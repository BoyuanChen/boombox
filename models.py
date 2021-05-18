

import os
import cv2
import glob
import torch
import shutil
import numpy as np
from torch import nn
from tqdm import tqdm
from PIL import Image
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from collections import OrderedDict
from torchvision.utils import save_image
from utils import MaskedMSELoss, MaskedL1Loss, DepthEvalResult
from utils import get_color_filtered_binary_mask_and_rect, get_iou_score_from_masks, get_segmented_binary_mask, get_object_rect_from_depth


from dataset import SoundBoxDataset
from model_utils import EncoderDecoder


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


class SoundBoxModel(pl.LightningModule):

    def __init__(self,
                 lr: float=1e-4,
                 seed: int=1,
                 if_cuda: bool=True,
                 if_test: bool=False,
                 gamma: float=0.5,
                 log_dir: str='logs',
                 train_batch: int=32,
                 val_batch: int=32,
                 test_batch: int=32,
                 num_workers: int=8,
                 in_channels: int=28,
                 model_name: str='conv2d-encoder-decoder',
                 num_branches: int=3,
                 branches_in_channels: list=[12, 4, 12],
                 data_filepath: str='data',
                 shapes: list=['cube'],
                 if_sound: bool=True,
                 if_vision: bool=True,
                 if_depth: bool=True,
                 if_all_input_data: bool=True,
                 depth_representation: str='image',
                 output_representation: str='pixel',
                 lr_schedule: list=[20, 50, 100],
                 test_hsv_threshold_lst: list=[[0, 13, 147], [42, 170, 255]]) -> None:
        super().__init__()
        self.lr = lr
        self.seed = seed
        self.if_cuda = if_cuda
        self.if_test = if_test
        self.gamma = gamma
        self.log_dir = os.path.join(log_dir, 'pred_visualizations')
        if not self.if_test:
            mkdir(self.log_dir)
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.test_batch = test_batch
        self.num_workers = num_workers
        self.in_channels = in_channels
        self.data_filepath = data_filepath
        self.shapes = shapes
        self.lr_schedule = lr_schedule
        self.model_name = model_name
        self.num_branches = num_branches
        self.branches_in_channels = branches_in_channels
        self.test_hsv_threshold_lst = test_hsv_threshold_lst
        self.kwargs = {'num_workers': self.num_workers, 'pin_memory': True} if self.if_cuda else {}

        self.if_sound = if_sound
        self.if_vision = if_vision
        self.if_depth = if_depth
        self.if_all_input_data = if_all_input_data
        self.depth_representation = depth_representation
        self.output_representation = output_representation

        if 'depth' in self.output_representation:
            self.depth_eval_result = DepthEvalResult()

        self.__build_model()

    def __build_model(self):
        # Model
        if self.model_name == 'conv2d-encoder-decoder':
            self.model = EncoderDecoder(in_channels=self.in_channels, output_representation=self.output_representation)

        # Loss
        if self.output_representation == 'segmentation':
            self.loss_func = nn.BCEWithLogitsLoss()
        if self.output_representation == 'depth-l1':
            self.loss_func = MaskedL1Loss()
        if self.output_representation == 'depth-l2':
            self.loss_func = MaskedMSELoss()
        if self.output_representation == 'pixel':
            self.loss_func = nn.MSELoss()

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)

        train_loss = self.loss_func(output, target)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target, path = batch
        output = self.model(data)

        val_loss = self.loss_func(output, target)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.output_representation == 'pixel' or self.output_representation == 'segmentation':
            iou_score, center_success = self.get_iou_score(output, path, mode=self.output_representation)
            self.log('iou_score', iou_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('center_success', center_success, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('num_of_samples', output.shape[0], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if 'depth' in self.output_representation:
            self.depth_eval_result.evaluate(output.data, target.data)
            self.log('rmse', self.depth_eval_result.rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('absrel', self.depth_eval_result.absrel, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('delta1', self.depth_eval_result.delta1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('delta2', self.depth_eval_result.delta2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('delta3', self.depth_eval_result.delta3, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('iou_score', self.depth_eval_result.iou_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('center_success', self.depth_eval_result.center_success, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('num_of_samples', output.shape[0], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        data, target, path = batch
        output = self.model(data)
        # get the loss evaluation score
        test_loss = self.loss_func(output, target)
        # get the evaluation score
        if self.output_representation == 'pixel' or self.output_representation == 'segmentation':
            iou_score, center_success = self.get_iou_score(output, path, mode=self.output_representation)
        if 'depth' in self.output_representation:
            self.depth_eval_result.evaluate(output.data, target.data)
        
        # plot the output and ground-truth images
        if self.output_representation == 'pixel' or self.output_representation == 'segmentation':
            comparison = torch.cat([output, target])
            if isinstance(data, list):
                device_num = data[0].get_device()
            else:
                device_num = data.get_device()
            save_image(comparison.cpu(), os.path.join(self.log_dir, str(device_num) + '_' + str(batch_idx) + '.png'), nrow=output.shape[0])
        if 'depth' in self.output_representation:
            if isinstance(data, list):
                device_num = data[0].get_device()
            else:
                device_num = data.get_device()
            output = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            for idx in range(target.shape[0]):
                color_output = cv2.applyColorMap(cv2.convertScaleAbs(output[idx][0] * 430.0, alpha=0.5), cv2.COLORMAP_TURBO)
                color_target = cv2.applyColorMap(cv2.convertScaleAbs(target[idx][0] * 430.0, alpha=0.5), cv2.COLORMAP_TURBO)
                images = np.hstack((color_target, color_output))
                # images = Image.fromarray(images)
                cv2.imwrite(os.path.join(self.log_dir, str(device_num) + '_' + str(batch_idx) + '_' + str(idx) + '.png'), images)

        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.output_representation == 'pixel' or self.output_representation == 'segmentation':
            self.log('iou_score', iou_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('center_success', center_success, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('num_of_samples', output.shape[0], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if 'depth' in self.output_representation:
            self.log('rmse', self.depth_eval_result.rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('absrel', self.depth_eval_result.absrel, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('delta1', self.depth_eval_result.delta1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('delta2', self.depth_eval_result.delta2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('delta3', self.depth_eval_result.delta3, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('center_success', self.depth_eval_result.center_success, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('iou_score', self.depth_eval_result.iou_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('num_of_samples', output.shape[0], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    # mode: pixel or segmentation
    def get_iou_score(self, output, target_paths, mode='pixel'):
        output = output.cpu().detach().numpy()
        iou_score_lst = []
        total_num = output.shape[0]
        success_count = 0.0
        if mode == 'pixel':
            for idx in range(total_num):
                output_mask_rect = get_color_filtered_binary_mask_and_rect(output[idx], self.test_hsv_threshold_lst)
                if len(output_mask_rect) != 2:
                    output_mask = output_mask_rect
                    output_rect = None
                else:
                    output_mask = output_mask_rect[0]
                    output_rect = output_mask_rect[1]
                
                target_depth_path = glob.glob(os.path.join(target_paths[idx], 'top_depth_*_cut.npy'))[0]
                target_depth = (np.load(target_depth_path)) / 430.0
                target_mask, target_rect = get_object_rect_from_depth(target_depth)
                iou_score = get_iou_score_from_masks(output_mask, target_mask)
                iou_score_lst.append(iou_score)
                # center success with rect
                if output_rect is not None:
                    dist = np.linalg.norm(np.array(output_rect[0]) - np.array(target_rect[0]))
                    target_max = max(target_rect[1])
                    if dist <= target_max:
                        success_count = success_count + 1.0
        if mode == 'segmentation':
            for idx in range(output.shape[0]):
                output_mask = get_segmented_binary_mask(output[idx])
                target_mask = target[idx][0]
                iou_score = get_iou_score_from_masks(output_mask, target_mask)
                iou_score_lst.append(iou_score)
        iou_score_lst = np.array(iou_score_lst)
        return np.mean(iou_score_lst), success_count

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_schedule, gamma=self.gamma)
        return [optimizer], [scheduler]


    def setup(self, stage=None):

        if stage == 'fit':
            self.train_dataset = SoundBoxDataset(self.data_filepath,
                                                 self.shapes,
                                                 flag='train',
                                                 transform=None,
                                                 seed=self.seed,
                                                 depth_representation=self.depth_representation,
                                                 if_vision=self.if_vision,
                                                 if_sound=self.if_sound,
                                                 if_depth=self.if_depth,
                                                 if_all_input_data=self.if_all_input_data,
                                                 output_representation=self.output_representation)

            self.val_dataset = SoundBoxDataset(self.data_filepath,
                                               self.shapes,
                                               flag='val',
                                               transform=None,
                                               seed=self.seed,
                                               depth_representation=self.depth_representation,
                                               if_vision=self.if_vision,
                                               if_sound=self.if_sound,
                                               if_depth=self.if_depth,
                                               if_all_input_data=self.if_all_input_data,
                                               output_representation=self.output_representation)
        
        if stage == 'test':
            self.test_dataset = SoundBoxDataset(self.data_filepath,
                                                self.shapes,
                                                flag='test',
                                                transform=None,
                                                seed=self.seed,
                                                depth_representation=self.depth_representation,
                                                if_vision=self.if_vision,
                                                if_sound=self.if_sound,
                                                if_depth=self.if_depth,
                                                if_all_input_data=self.if_all_input_data,
                                                output_representation=self.output_representation)
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.train_batch,
                                                   shuffle=True,
                                                   **self.kwargs)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                 batch_size=self.val_batch,
                                                 shuffle=False,
                                                 **self.kwargs)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.test_batch,
                                                  shuffle=False,
                                                  **self.kwargs)
        return test_loader
