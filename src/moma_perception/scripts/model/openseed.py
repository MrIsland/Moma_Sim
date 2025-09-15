import os, sys
from PIL import Image
import numpy as np
import torch
from tqdm import trange, tqdm
import gzip

from torchvision import transforms
from utils.openseed import *
from utils.openseed.BaseModel import *
from utils.openseed.colormap import *
from utils.openseed.catalog import *
from utils.openseed.utils.visualizer import Visualizer

from utils.common import *


class OpenSeed():
    def __init__(self, cfg):
        super(OpenSeed, self).__init__()
        self.cfg = cfg
        self.project_name = self.cfg['semantic']['project_name']
        self.pretrained_pth = self.cfg['semantic']['pretrained_path']
        self.pretrained_model = os.path.join(self.pretrained_pth, self.cfg['semantic']['pretrained_model'])
        opt = self.load_opt_command(self.cfg)
        # self.model = BaseModel(opt, build_model(opt)).eval().cuda()
        self.model = BaseModel(opt, build_model(opt)).from_pretrained(self.pretrained_model).eval().cuda()
        t = []
        t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
        self.transform = transforms.Compose(t)
        self.thing_classes = []
        with open(os.path.join(self.cfg['semantic']['semantic_config_path'], self.project_name, self.cfg['semantic']['class_txt']), 'r') as file:
            for line in file:
                self.thing_classes.append(line.strip())
        stuff_classes = []
        self.thing_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(self.thing_classes))]
        stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
        thing_dataset_id_to_contiguous_id = {x: x for x in range(len(self.thing_classes))}
        stuff_dataset_id_to_contiguous_id = {x + len(self.thing_classes): x for x in range(len(stuff_classes))}

        MetadataCatalog.get("demo").set(
            thing_colors=self.thing_colors,
            thing_classes=self.thing_classes,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
            stuff_colors=stuff_colors,
            stuff_classes=stuff_classes,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        )

        self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(self.thing_classes + stuff_classes,
                                                                            is_eval=False)

        metadata = MetadataCatalog.get('demo')
        self.model.model.metadata = metadata
        self.model.model.sem_seg_head.num_classes = len(self.thing_classes + stuff_classes)
        self.slam_cfg = load_config(os.path.join(self.cfg['semantic']['slam_config_path'], self.cfg['semantic']['slam_config_yaml']))

        self.output_dir = os.path.join(self.cfg['project_info']['output_path'], self.project_name)
        self.detection_path = os.path.join(self.output_dir, 'detection')
        self.visualization_path = os.path.join(self.output_dir, 'visual')
        self.pkl_pth = os.path.join(self.output_dir, 'pkl')
        self.pcd_path = os.path.join(self.output_dir, 'pcd')
        self.pcd_save_path = os.path.join(self.output_dir, 'pcd_save')
        self.som_path = os.path.join(self.output_dir, 'som')

        os.makedirs(self.detection_path, exist_ok=True)
        os.makedirs(self.visualization_path, exist_ok=True)
        os.makedirs(self.pkl_pth, exist_ok=True)
        os.makedirs(self.pcd_path, exist_ok=True)
        os.makedirs(self.pcd_save_path, exist_ok=True)
        os.makedirs(self.som_path, exist_ok=True)

    def load_opt_command(self, cfg):
        opt = load_opt_from_config_files([cfg['semantic']['swinl_lang_decouple_yaml_path']])
        overrides=['WEIGHT', os.path.join(cfg['semantic']['pretrained_path'], cfg['semantic']['pretrained_model'])]
        if overrides:
            assert len(overrides) % 2 == 0, "overrides arguments is not paired, required: key value"
            keys = [overrides[idx * 2] for idx in range(len(overrides) // 2)]
            vals = [overrides[idx * 2 + 1] for idx in range(len(overrides) // 2)]
            vals = [val.replace('false', '').replace('False', '') if len(val.replace(' ', '')) == 5 else val for val in
                    vals]

            types = []
            for key in keys:
                key = key.split('.')
                ele = opt.copy()
                while len(key) > 0:
                    ele = ele[key.pop(0)]
                types.append(type(ele))

            config_dict = {x: z(y) for x, y, z in zip(keys, vals, types)}
            load_config_dict_to_opt(opt, config_dict)

        return opt

    def convertMask(self, mask_origin, ids):
        label_length=len(ids)
        masks=np.zeros((label_length,mask_origin.shape[0],mask_origin.shape[1]),dtype=np.bool_)
        for i in range(label_length) :
            masks[i]=(mask_origin == ids[i])
        return masks

    def detection(self, color_img, idx, visual_save_path=None, detection_save_path=None):
        with torch.no_grad():
            # image_ori = Image.open(color_img_path).convert("RGB")
            # print('color_img:   {}'.format(color_img))
            image_ori = Image.fromarray(color_img.astype(np.uint8))
            width = image_ori.size[0]
            height = image_ori.size[1]
            image = self.transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2, 0, 1).cuda()
            batch_inputs = [{'image': images, 'height': height, 'width': width}]
            outputs = self.model.forward(batch_inputs)
            visual = Visualizer(image_ori, metadata=self.model.model.metadata)
            pano_seg = outputs[-1]['panoptic_seg'][0]
            pano_seg_info = outputs[-1]['panoptic_seg'][1]

            for i in range(len(pano_seg_info)):
                if pano_seg_info[i]['category_id'] in self.model.model.metadata.thing_dataset_id_to_contiguous_id.keys():
                    pano_seg_info[i]['category_id'] = self.model.model.metadata.thing_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                    pano_seg_info[i]['category_name'] = self.model.model.metadata.thing_classes[pano_seg_info[i]['category_id']]
                else:
                    pano_seg_info[i]['isthing'] = False
                    pano_seg_info[i]['category_id'] = self.model.model.metadata.stuff_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                    pano_seg_info[i]['category_name'] = self.model.model.metadata.thing_classes[pano_seg_info[i]['category_id']]

            ### visualization
            # demo.save(os.path.join(visual_save_path, str(color_img_path).split('/')[-1][:-4] + '.png'))
            if visual_save_path is not None:
                demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info)  # rgb Image
                demo.save(os.path.join(visual_save_path, str(idx) + '.png'))

            ### save result
            id = [pano['id'] for pano in pano_seg_info]
            mask = self.convertMask(pano_seg.cpu().numpy(),id)
            labels = [self.model.model.metadata.thing_classes[pano['category_id']] for pano in pano_seg_info]
            if len(labels) > 0:
                visual_feature = torch.stack([pano['semantic_feature'] for pano in pano_seg_info])
                # bbox = torch.stack([pano['bbox'] for pano in pano_seg_info])
                category_id = [pano['category_id'] for pano in pano_seg_info]
                category_name = [pano['category_name'] for pano in pano_seg_info]
                # visual_feature = torch.stack([pano['semantic_feature'] for pano in pano_seg_info])
                # text_feature = torch.stack([getattr(self.model.model.sem_seg_head.predictor.lang_encoder,
                                                    # 'default_text_embeddings')[pano['category_id']] for pano in pano_seg_info])
                if len(category_id) > 0:
                    results = {
                        # "xyxy": bbox.cpu().numpy(),
                        "class_id": np.asarray(category_id),
                        "mask": mask,
                        "classes": category_name,
                        # "image_feats": visual_feature.cpu().numpy(),
                        # "text_feats": text_feature.cpu().numpy(),
                    }
                    if detection_save_path is not None:
                        # with gzip.open(os.path.join(detection_save_path, str(color_img_path).split('/')[-1][:-4] + ".pkl.gz"), "wb") as f:
                        with gzip.open(os.path.join(detection_save_path, str(idx) + ".pkl.gz"), "wb") as f:
                            pkl.dump(results, f)
            else:
                results = None

        return results