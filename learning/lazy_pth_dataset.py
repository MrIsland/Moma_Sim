import os, sys
import pdb

import torch
from torch.utils.data import Dataset, DataLoader

class LazyPthDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pth')],
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )

        # self.data = []
        # for file in self.files:
        #     content = torch.load(file, map_location='cpu')
        #     if isinstance(content, list):  # 如果每个文件是list
        #         self.data.extend(content)
        #     elif isinstance(content, dict):  # 如果是单个dict
        #         self.data.append(content)
        #     else:
        #         raise ValueError(f"Unsupported type in {file}: {type(content)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        sample = torch.load(file_path, map_location='cpu')

        save_dict_idx = sample['save_dict_idx']
        num_envs = sample['num_envs']
        obs = sample['obs']
        action = sample['action']

        return sample

if __name__ == '__main__':
    dataset = LazyPthDataset('/media/island/igrape_20T/island/moma/vision_trained_data/1111')
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for batch in loader:
        print('save_dict_idx:   {}; num_envs:   {}'.format(batch['save_dict_idx'], batch['num_envs']))
        print('??????????????')