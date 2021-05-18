
import os
import sys
import json
import glob
import torch
import itertools
import numpy as np
from PIL import Image
from scipy import misc
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class SoundBoxDataset(Dataset):
    def __init__(self,
                 data_filepath,
                 shapes,
                 flag,
                 transform=None,
                 seed=1,
                 depth_representation='array',
                 if_vision=True,
                 if_sound=True,
                 if_depth=True,
                 if_all_input_data=True,
                 output_representation='pixel'):

        self.data_filepath = data_filepath
        self.shapes = shapes
        self.flag = flag
        self.transform = transform
        self.seed = seed
        self.ratio = 0.8
        self.if_vision = if_vision
        self.if_sound = if_sound
        self.if_depth = if_depth
        self.if_all_input_data = if_all_input_data
        self.depth_representation = depth_representation # 'array' or 'image'
        self.output_representation = output_representation # 'pixel' or 'segmentation' or 'depth-l1' or 'depth-l2'
        self.all_sequences = self.get_all_sequences()
    
    def get_all_sequences(self):
        all_sequences = []
        for p_shape in self.shapes:
            # get the splits from the original folder, even if for ablation studies
            if 'cube' in p_shape:
                tmp_p_shape = 'cube'
            if 'small_cuboid' in p_shape:
                tmp_p_shape = 'small_cuboid'
            if 'large_cuboid' in p_shape:
                tmp_p_shape = 'large_cuboid'

            with open(os.path.join(self.data_filepath, tmp_p_shape, f'data_split_dict_{self.seed}.json'), 'r') as file:
                seq_dict = json.load(file)
            tmp_sequences = seq_dict[self.flag]
            for idx in range(len(tmp_sequences)):
                tmp_sequences[idx] = tmp_sequences[idx].replace(tmp_p_shape, p_shape)
                tmp_sequences[idx] = os.path.join(self.data_filepath, tmp_sequences[idx][1:])
            all_sequences.append(tmp_sequences)
        all_sequences = list(itertools.chain.from_iterable(all_sequences))
        return all_sequences

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        # load audio
        track0_spec_filepath = self.all_sequences[idx]
        selected_shape = track0_spec_filepath.split('/')[-5]
        selected_day = track0_spec_filepath.split('/')[-3]
        selected_seq = track0_spec_filepath.split('/')[-1]
        track0_spec = np.load(track0_spec_filepath) / -80.
        track1_spec = np.load(os.path.join(self.data_filepath, selected_shape, 'audio', selected_day, '1', selected_seq)) / -80.
        track2_spec = np.load(os.path.join(self.data_filepath, selected_shape, 'audio', selected_day, '2', selected_seq)) / -80.
        track3_spec = np.load(os.path.join(self.data_filepath, selected_shape, 'audio', selected_day, '3', selected_seq)) / -80.
        track0_spec = torch.tensor(track0_spec).unsqueeze(0)
        track1_spec = torch.tensor(track1_spec).unsqueeze(0)
        track2_spec = torch.tensor(track2_spec).unsqueeze(0)
        track3_spec = torch.tensor(track3_spec).unsqueeze(0)
        if self.if_all_input_data:
            audio = [track0_spec,
                     track1_spec,
                     track2_spec,
                     track3_spec]
        else:
            audio = [track0_spec.unsqueeze(0),
                     track1_spec.unsqueeze(0),
                     track2_spec.unsqueeze(0),
                     track3_spec.unsqueeze(0)]
        
        # load image
        # get video from the original folder, even for ablation studies
        if 'cube' in selected_shape:
            selected_shape = 'cube'
        if 'small_cuboid' in selected_shape:
            selected_shape = 'small_cuboid'
        if 'large_cuboid' in selected_shape:
            selected_shape = 'large_cuboid'

        selected_seq_idx = selected_seq.split('.')[0].split('_')[1]
        img_filepaths = glob.glob(os.path.join(self.data_filepath, selected_shape, 'video', selected_day, selected_seq_idx, 'side_depth_*.npy'))
        ids = []
        for p_path in img_filepaths:
            ids.append(str(p_path.split('/')[-1].split('.')[0].split('_')[-1]))
        ids.sort()
        rgb_imgs = []
        depth_imgs = []
        for p_id in ids:
            rgb_img = np.array(Image.open(os.path.join(self.data_filepath, selected_shape, 'video', selected_day, selected_seq_idx, f'side_rgb_{p_id}.png'))) / 255.0
            rgb_img = torch.tensor(rgb_img)
            rgb_img = rgb_img.permute(2, 0, 1)
            if self.if_all_input_data:
                rgb_imgs.append(rgb_img)
            else:
                rgb_imgs.append(rgb_img.unsqueeze(0))

            if self.depth_representation == 'array':
                depth_img = np.load(os.path.join(self.data_filepath, selected_shape, 'video', selected_day, selected_seq_idx, f'side_depth_{p_id}.npy'))
                depth_img = torch.tensor(depth_img)
                depth_img = depth_img.unsqueeze(0)
            elif self.depth_representation == 'array_normalized':
                depth_img = np.load(os.path.join(self.data_filepath, selected_shape, 'video', selected_day, selected_seq_idx, f'side_depth_{p_id}.npy'))
                depth_img = (depth_img + 148.63086) / (2398.7593 + 148.63086) #TODO
                depth_img = torch.tensor(depth_img)
                depth_img = depth_img.unsqueeze(0)
            else:
                depth_img = np.array(Image.open(os.path.join(self.data_filepath, selected_shape, 'video', selected_day, selected_seq_idx, f'side_depth_{p_id}.png'))) / 255.0
                depth_img = torch.tensor(depth_img)
                depth_img = depth_img.permute(2, 0, 1)
            if self.if_all_input_data:
                depth_imgs.append(depth_img)
            else:
                depth_imgs.append(depth_img.unsqueeze(0))

        # load target
        if self.output_representation == 'pixel':
            target_filepath = glob.glob(os.path.join(self.data_filepath, selected_shape, 'video', selected_day, selected_seq_idx, 'top_rgb_*.png'))[0]
            target = np.array(Image.open(target_filepath)) / 255.0
        if self.output_representation == 'segmentation':
            target_filepath = glob.glob(os.path.join(self.data_filepath, selected_shape, 'video', selected_day, selected_seq_idx, 'top_segmentation_*.png'))[0]
            target = Image.open(target_filepath).convert('1')
            target = np.array(target).astype('uint8')
            target = np.expand_dims(target, 2)
        if 'depth' in  self.output_representation:
            target_filepath = glob.glob(os.path.join(self.data_filepath, selected_shape, 'video', selected_day, selected_seq_idx, 'top_depth_*_cut.npy'))[0]
            target = np.load(target_filepath)
            target = target / 430.0
            target = np.expand_dims(target, 2)

        # to tensor
        audio = torch.cat(audio, axis=0).float()
        rgb_imgs = torch.cat(rgb_imgs, axis=0)
        depth_imgs = torch.cat(depth_imgs, axis=0)
        target = torch.tensor(target)
        target = target.permute(2, 0, 1).float()

        ######## All Combinations ########
        # vision (12)
        if self.if_vision and not self.if_sound and not self.if_depth:
            rgb_imgs = rgb_imgs.float()
            audio = None
            depth_imgs = None
        # vision, sound (12 + 4 = 16)
        if self.if_vision and self.if_sound and not self.if_depth:
            rgb_imgs = rgb_imgs.float()
            audio = audio
            depth_imgs = None
        # vision, depth
        if self.if_vision and not self.if_sound and self.if_depth:
            if 'array' in self.depth_representation:
                # (12 + 4 = 16)
                rgb_imgs = rgb_imgs.float()
                audio = None
                depth_imgs = depth_imgs
            else:
                # (12 + 12 = 24)
                rgb_imgs = rgb_imgs.float()
                audio = None
                depth_imgs = depth_imgs.float()
        # vision, sound, depth
        if self.if_vision and self.if_sound and self.if_depth:
            if 'array' in self.depth_representation:
                # (12 + 4 + 4 = 20)
                rgb_imgs = rgb_imgs.float()
                audio = audio
                depth_imgs = depth_imgs
            else:
                # (12 + 4 + 12 = 28)
                rgb_imgs = rgb_imgs.float()
                audio = audio
                depth_imgs = depth_imgs.float()
        # sound (4)
        if not self.if_vision and self.if_sound and not self.if_depth:
            rgb_imgs = None
            audio = audio
            depth_imgs = None
        # depth
        if not self.if_vision and not self.if_sound and self.if_depth:
            if 'array' in self.depth_representation:
                # (4)
                rgb_imgs = None
                audio = None
                depth_imgs = depth_imgs
            else:
                # (12)
                rgb_imgs = None
                audio = None
                depth_imgs = depth_imgs.float()
        # sound, depth
        if not self.if_vision and self.if_sound and self.if_depth:
            if 'array' in self.depth_representation:
                # (4 + 4 = 8)
                rgb_imgs = None
                audio = audio
                depth_imgs = depth_imgs
            else:
                # (4 + 12 = 16)
                rgb_imgs = None
                audio = audio
                depth_imgs = depth_imgs.float()
        
        input_data_list = []
        if audio is not None:
            input_data_list.append(audio)
        if rgb_imgs is not None:
            input_data_list.append(rgb_imgs)
        if depth_imgs is not None:
            input_data_list.append(depth_imgs)

        if self.if_all_input_data:
            input_data = torch.cat(input_data_list, axis=0)
            if self.flag == 'test' or self.flag == 'val':
                return input_data, target, os.path.join(self.data_filepath, selected_shape, 'video', selected_day, selected_seq_idx)
            else:
                return input_data, target
        else:
            if self.flag == 'test' or self.flag == 'val':
                return input_data_list, target, os.path.join(self.data_filepath, selected_shape, 'video', selected_day, selected_seq_idx)
            else:
                return input_data_list, target
