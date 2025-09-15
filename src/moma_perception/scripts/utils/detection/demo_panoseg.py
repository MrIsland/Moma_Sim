# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)

import torch
from torchvision import transforms
from tqdm import trange
from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.visualizer import Visualizer


# logger = logging.getLogger(__name__)

# def initModel():


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt = load_opt_command()
    pretrained_pth = "/home/caoyihan/Downloads/openseed_swinl_pano_sota.pt"
    image_dir = '/data/Dataset/'
    image_pth = 'SGG/image/'
    output_root = './output/'+image_pth+'panoseg_bbox/'
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
    thing_classes=[]
    with open('configs/panoptic_categories_nomerge.txt', 'r') as file:
        for line in file:
                thing_classes.append(line.strip())
    # thing_classes = ['car','person','traffic light', 'truck', 'motorcycle']
    stuff_classes = ["wall", "floor", "ceiling"]
    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(thing_classes))]
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
    stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
    
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)
    image_name_list = os.listdir(image_dir+image_pth)
    # new=[item for item in image_name_list if "frame" in item]
    # image_name_list = new
    image_name_list.sort()
    print(len(image_name_list))
    for j in trange(len(image_name_list)):
        # if j % 5 == 0 :
        with torch.no_grad():
            image_ori = Image.open(image_dir+image_pth+image_name_list[j]).convert("RGB")
            width = image_ori.size[0]
            height = image_ori.size[1]
            image = transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

            batch_inputs = [{'image': images, 'height': height, 'width': width}]
            outputs = model.forward(batch_inputs)
            visual = Visualizer(image_ori, metadata=metadata)
            pano_seg = outputs[-1]['panoptic_seg'][0]
            pano_seg_info = outputs[-1]['panoptic_seg'][1]
            
            for i in range(len(pano_seg_info)):
                if pano_seg_info[i]['category_id'] in metadata.thing_dataset_id_to_contiguous_id.keys():
                    pano_seg_info[i]['category_id'] = metadata.thing_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                    pano_seg_info[i]['category_name'] = metadata.thing_classes[pano_seg_info[i]['category_id']]
                else:
                    pano_seg_info[i]['isthing'] = False
                    pano_seg_info[i]['category_id'] = metadata.stuff_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                    pano_seg_info[i]['category_name'] = metadata.thing_classes[pano_seg_info[i]['category_id']]
            labels = [metadata.thing_classes[pano['category_id']] for pano in pano_seg_info]
            feature_tensor=torch.stack([pano['semantic_feature'] for pano in pano_seg_info])
            demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image
            demo.save(os.path.join(output_root, image_name_list[j][:-4]+'.png'))


if __name__ == "__main__":
    main()
    sys.exit(0)