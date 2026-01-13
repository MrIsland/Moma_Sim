import os, sys
import pdb

import cv2
import glob
import torch
import argparse
import matplotlib
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from learning.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from tqdm import trange
import matplotlib.pyplot as plt


class ReduceAndPool(nn.Module):
    def __init__(self, in_ch=128, proj_ch=32, out_spatial=(1, 1)):
        super(ReduceAndPool, self).__init__()
        self.proj = nn.Conv2d(in_ch, proj_ch, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(out_spatial)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(proj_ch * out_spatial[0] * out_spatial[1], 256),
            nn.ReLU()
        )

    def forward(self, feats):
        # feats: (B, C, H, W)
        # pdb.set_trace()
        x = self.proj(feats)
        x = self.pool(x)
        x = self.mlp(x)
        return x


class DepthAnythingV2Encoder(nn.Module):
    def __init__(self, encoder, device):
        super(DepthAnythingV2Encoder, self).__init__()
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.device = device
        # self.input_size = input_size
        self.depth_anything = DepthAnythingV2(**model_configs[encoder])
        self.depth_anything.load_state_dict(
            torch.load(f'learning/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu')
            # torch.load(f'DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu')
        )
        self.depth_model = self.depth_anything.to(device).eval()
        # self.reduce_and_pool = ReduceAndPool(in_ch=64, proj_ch=32, out_spatial=(1, 1)).to(device)

    @torch.no_grad()
    def forward_batch(self, image_batch):
        if image_batch.max() > 1.0:
            image_batch /= 255.0
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        mean_t = torch.tensor(mean, device=image_batch.device, dtype=image_batch.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=image_batch.device, dtype=image_batch.dtype).view(1, 3, 1, 1)
        tensor = (image_batch - mean_t) / std_t  # 函数 PrepareForNet 本身已是 (C,H,W)，这里不需要再转置

        image_tensors = tensor.to(self.device)
        num_img_batch = image_batch.shape[0]
        partial_img_feats_list = []
        step = min(num_img_batch, 64)
        for i in range(0, num_img_batch, step):
            partial_img_tensor_batch = image_tensors[i:i+step]
            outputs, patch_feature_list, dino_feature = self.depth_model(partial_img_tensor_batch)
            depth_feats = patch_feature_list[-1].permute(0, 2, 3, 1)
            partial_img_feats_list.append(depth_feats)
            # DEBUG_visualize(image_batch, outputs)
        partial_img_feats_batch = torch.stack(partial_img_feats_list)
        partial_img_feats_batch = partial_img_feats_batch.reshape(-1, *partial_img_feats_batch.shape[2:])
        return partial_img_feats_batch   # (batch, 16, 16, 64)


    def forward(self, image_batch):
        feats = self.forward_batch(image_batch.permute(0, 3, 1, 2))
        # feats = self.reduce_and_pool(feats.permute(0, 3, 1, 2))
        return feats
        # print('Visual Encoder - DA ')

def pre_process_rgb_img(img_batch, img_h, img_w):
    img_batch = img_batch.permute(0, 3, 1, 2)

    _, _, h, w = img_batch.shape
    left = (w - h) // 2
    img_batch = img_batch[:, :, :, left:left+w]
    img_batch = F.interpolate(img_batch, size=(img_h, img_w), mode='bilinear', align_corners=False)

    img_batch = img_batch.permute(0, 2, 3, 1)
    return img_batch

def DEBUG_visualize(rgb_batch, depth_batch, path='/home/island/Desktop/mobile_manipulation/IsaacGymEnvs/isaacgymenvs/test_vis'):
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    rgb_batch = rgb_batch.permute(0, 2, 3, 1)
    import matplotlib.pyplot as plt
    for i, depth in enumerate(depth_batch):
        final_path = os.path.join(path, '{}'.format(i))
        depth_img = os.path.join(final_path, 'depth_anything.png')
        depth_image = depth.cpu().numpy()
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255.0
        depth_image = depth_image.astype(np.uint8)
        depth_image = (cmap(depth_image)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        cv2.imwrite(depth_img, depth_image)

        rgb_img = os.path.join(final_path, 'rgb.png')
        rgb_image = rgb_batch[i].cpu().numpy() * 255.
        cv2.imwrite(rgb_img, rgb_image)


if __name__ == '__main__':
    img_path = '/home/island/Desktop/mobile_manipulation/IsaacGymEnvs/isaacgymenvs/test_vis/0/rgb.png'
    visual_encoder = DepthAnythingV2Encoder(encoder='vits', device='cuda', input_size=448)
    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)
    # filenames = filenames[:128]

    image_list = []
    for k, filename in enumerate(filenames):
        raw_image = cv2.imread(filename)
        image_list.append(raw_image)
    image_batch = torch.from_numpy(np.stack(image_list))    # (B, H, W, 3)
    # image_batch = pre_process_rgb_img(image_batch)
    if image_batch.max() > 1.0:
        image_batch = image_batch / 255.0
    image_tensors = image_batch.to('cuda')

    isaac_tensor_path = '/home/island/Desktop/mobile_manipulation/IsaacGymEnvs/isaacgymenvs/test_vis/0/partial1.pt'
    isaac_tensor = torch.load(isaac_tensor_path)

    with torch.no_grad():
        feats = visual_encoder(image_tensors)
    # DEBUG_visualize(image_tensors, depth)
    # plt.imshow(depth[0].detach().cpu().numpy(), cmap='gray')
    # plt.show()
    # visual_feats = visual_encoder.forward(image_batch)
    print('depth_anything_v2_encoder')


