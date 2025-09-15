'''
The script is used to extract Grounded SAM results on a posed RGB-D dataset. 
The results will be dumped to a folder under the scene folder. 
'''

import os
from typing import Any, List
from PIL import Image
import cv2
import json
import imageio
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import pickle
import gzip
import open_clip

import torch
import torchvision
from torch.utils.data import Dataset
import supervision as sv
from tqdm import trange

from dataset.datasets_common import get_dataset
from utils.vis import vis_result_fast, vis_result_slow_caption
from utils.model_utils import compute_clip_features
import torch.nn.functional as F


#Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo

GSA_PATH = "/home/caoyihan/Code/NAVIGATION/OpenFMNav_dependencies/Grounded-Segment-Anything/"
    
try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

import torchvision.transforms as TS
try:
    from ram.models import ram
    from ram.models import tag2text
    from ram import inference_tag2text, inference_ram
except ImportError as e:
    print("RAM sub-package not found. Please check your GSA_PATH. ")
    raise e

# Disable torch gradient computation
torch.set_grad_enabled(False)
import sys
TAG2TEXT_PATH = os.path.join(GSA_PATH, "")
EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
sys.path.append(EFFICIENTSAM_PATH)

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

FOREGROUND_GENERIC_CLASSES = [
    "item", "furniture", "object", "electronics", "wall decoration", "door"
]

FOREGROUND_MINIMAL_CLASSES = [
    "item"
]


# Prompting SAM with detected boxes
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def get_sam_predictor(variant: str, device) -> SamPredictor:
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor
    

def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> List[str]:
    '''
    Convert a text prompt from Tag2Text to a list of classes. 
    '''
    classes = text_prompt.split(',')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes

def InitDetection(args):
    ### Initialize the Grounding DINO model ###
    global tagging_model,tagging_transform,grounding_dino_model,sam_predictor,clip_model,clip_preprocess,clip_tokenizer
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
        device=args.device
    )

    ### Initialize the SAM model ###
    sam_predictor = get_sam_predictor(args.sam_variant, args.device)
    
    ###
    # Initialize the CLIP model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    # Initialize the dataset
    if args.class_set == "tag2text":
        # The class set will be computed by tag2text on each image
        # filter out attributes and action categories which are difficult to grounding
        delete_tag_index = []
        for i in range(3012, 3429):
            delete_tag_index.append(i)

        # load model
        tagging_model = tag2text.tag2text_caption(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                                                image_size=384,
                                                vit='swin_b',
                                                delete_tag_index=delete_tag_index)
        # threshold for tagging
        # we reduce the threshold to obtain more tags
        tagging_model.threshold = 0.64 
    elif args.class_set == "ram":
        tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                                        image_size=384,
                                        vit='swin_l')
        
    tagging_model = tagging_model.eval().to(args.device)
    
    # initialize Tag2Text
    tagging_transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(), 
        TS.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    return tagging_model,tagging_transform,grounding_dino_model,sam_predictor
def detection(cfg,color_path,global_classes,detections_save_path,vis_save_path):
    image = cv2.imread(color_path) # This will in BGR color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB color space
    image_pil = Image.fromarray(image_rgb)
    raw_image = image_pil.resize((384, 384))
    raw_image = tagging_transform(raw_image).unsqueeze(0).to(cfg.device)
    
    if cfg.class_set == "ram":
        res = inference_ram(raw_image , tagging_model)
        caption="NA"
    elif cfg.class_set == "tag2text":
        res = inference_tag2text.inference(raw_image , tagging_model, None)
        caption=res[2]

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    text_prompt=res[0].replace(' |', ',')
    
    # Add "other item" to capture objects not in the tag2text captions. 
    # Remove "xxx room", otherwise it will simply include the entire image
    # Also hide "wall" and "floor" for now...
    add_classes = ["other item"]
    remove_classes = [
        "room", "kitchen", "office", "house", "home", "building", "corner",
        "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
        "apartment", "image", "city", "blue", "skylight", "hallway", 
        "bureau", "modern", "salon", "doorway", "wall lamp"
    ]
    bg_classes = ["wall", "floor", "ceiling"]

    ## add bg classes
    add_classes += bg_classes

    classes = process_tag_classes(
        text_prompt, 
        add_classes = add_classes,
        remove_classes = remove_classes,
    )
        
    # add classes to global classes
    global_classes.update(classes)
    
    ## accumulate classes
    classes = list(global_classes)
        
    # Using GroundingDINO to detect and SAM to segment
    detections = grounding_dino_model.predict_with_classes(
        image=image, # This function expects a BGR image...
        classes=classes,
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
    )


    if len(detections.class_id) > 0:
        ### Non-maximum suppression ###
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            cfg.nms_threshold
        ).numpy().tolist()
        # print(f"After NMS: {len(detections.xyxy)} boxes")

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        
        # Somehow some detections will have class_id=-1, remove them
        valid_idx = detections.class_id != -1
        detections.xyxy = detections.xyxy[valid_idx]
        detections.confidence = detections.confidence[valid_idx]
        detections.class_id = detections.class_id[valid_idx]
        
        ### Segment Anything ###
        detections.mask = get_sam_segmentation_from_xyxy(
            sam_predictor=sam_predictor,
            image=image_rgb,
            xyxy=detections.xyxy
        )

        # Compute and save the clip features of detections  
        image_crops, image_feats, text_feats = compute_clip_features(
            image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, cfg.device)
    
    ### Visualize results ###
    annotated_image, labels = vis_result_fast(image, detections, classes)
    
    # save the annotated grounded-sam image
    # if cfg.class_set in ["ram", "tag2text"] and cfg.use_slow_vis:
    #     annotated_image_caption = vis_result_slow_caption(
    #         image_rgb, detections.mask, detections.xyxy, labels, caption, text_prompt)
    #     Image.fromarray(annotated_image_caption).save(vis_save_path)
    # else:
    cv2.imwrite(vis_save_path, annotated_image)
    
    
    # Convert the detections to a dict. The elements are in np.array
    results = {
        "xyxy": detections.xyxy,
        "confidence": detections.confidence,
        "class_id": detections.class_id,
        "mask": detections.mask,
        "classes": classes,
        "image_crops": image_crops,
        "image_feats": image_feats,
        "text_feats": text_feats,
    }
    
    if cfg.class_set in ["ram", "tag2text"]:
        results["tagging_caption"] = caption
        results["tagging_text_prompt"] = text_prompt
    

        
    # save the detections using pickle
    # Here we use gzip to compress the file, which could reduce the file size by 500x
    # with gzip.open(detections_save_path, "wb") as f:
    #     pickle.dump(results, f)
    
    # # save global classes
    # with open(args.dataset_root / args.scene_id / f"gsa_classes_{save_name}.json", "w") as f:
    #     json.dump(list(global_classes), f)
            
    # if args.save_video:
    #     imageio.mimsave(video_save_path, frames, fps=10)
    #     print(f"Video saved to {video_save_path}")
        
    return results