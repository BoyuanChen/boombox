
import os
import cv2
import glob
import json
import pprint
import random
import imutils
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def get_event_info(event_path):
    event = EventAccumulator(event_path)
    event.Reload()
    return event
    # print(event.Tags())
    # event.Scalars('val_loss_epoch')

def get_color_filtered_binary_mask_and_rect(img, test_hsv_threshold_lst):
    # Threshold
    lower = tuple(test_hsv_threshold_lst[0])
    upper = tuple(test_hsv_threshold_lst[1])

    # Tranpose image from pytorch to numpy
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = img * 255
    img = img.astype('uint8')

    # Get the binary mask template ready
    binary_mask = np.zeros((img.shape[1], img.shape[0], 1), np.uint8)

    # Find the largest contour under the color filtering range
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # If no contour found, return a black image
    if len(cnts) == 0:
        return binary_mask
    c = max(cnts, key=cv2.contourArea)

    # Draw the contour area on the mask and return it for iou computation
    cv2.drawContours(binary_mask, [c], -1, (255, 255, 255), -1)

    # Fit rect
    rect = cv2.minAreaRect(c)

    return binary_mask, rect

def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z

def get_segmented_binary_mask(img):
    img = img.squeeze(0)
    img = sigmoid(img)
    img[img>=0.5] = 1
    img[img<0.5] = 0
    img = img.astype('uint8')
    return img


def get_iou_score_from_masks(output_mask, target_mask):
    intersection = np.logical_and(output_mask, target_mask)
    union = np.logical_or(output_mask, target_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    if np.isnan(iou_score):
        assert False, 'iou_score has nan ...'
    return iou_score


# Uncomment the commented lines if you want to see the overlay images with certain number of samples instead of all of them
def get_average_last_top_rgb_imgs(data_filepath='/local/vondrick/bo/soundbox_learn/data/cube'):
    average_last_top_rgb_img = None
    count = 0.
    for idx in tqdm(range(5)):
        folders = glob.glob(os.path.join(data_filepath, 'video', str(idx), '*'))
        for p_folder in folders:
            img = np.array(Image.open(glob.glob(os.path.join(p_folder, 'top_rgb_*.png'))[0]), dtype=np.float)
            if average_last_top_rgb_img is None:
                average_last_top_rgb_img = img
            else:
                average_last_top_rgb_img = average_last_top_rgb_img + img
            count = count + 1
        #     if count == 3:
        #         break
        # if count == 3:
        #     break
    average_last_top_rgb_img = average_last_top_rgb_img / count
    average_last_top_rgb_img = np.array(np.round(average_last_top_rgb_img), dtype=np.uint8)
    # Get the shape information
    shape = data_filepath.split('/')[-1]
    out = Image.fromarray(average_last_top_rgb_img, mode='RGB')
    out.save(f'average_{shape}.png')


from PIL import Image
from PIL import ImageChops

# depth: numpy array representation, after de-normalized
def get_object_rect_from_depth(depth, lightning_variation_threshold=24):
    background_depth_img = np.load('background_depth.npy')
    background_depth_img = cv2.applyColorMap(cv2.convertScaleAbs(background_depth_img, alpha=0.5), cv2.COLORMAP_TURBO)
    background_depth_img = cv2.cvtColor(background_depth_img, cv2.COLOR_BGR2RGB)
    background_depth_img = Image.fromarray(background_depth_img)

    # convert depth image to color coded image
    if depth.shape[0] == 1:
        depth = cv2.applyColorMap(cv2.convertScaleAbs(depth[0] * 430.0, alpha=0.5), cv2.COLORMAP_TURBO)
    else:
        depth = cv2.applyColorMap(cv2.convertScaleAbs(depth * 430.0, alpha=0.5), cv2.COLORMAP_TURBO)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
    depth_img = Image.fromarray(depth)
    image = ImageChops.subtract(depth_img, background_depth_img)
    mask1 = Image.eval(image, lambda a: 0 if a <=24 else 255)
    mask2 = mask1.convert('1')
    mask2 = np.array(mask2).astype('uint8')

    cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) < 1:
        return None
    
    area_array = []
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        area_array.append(area)
    sorteddata = sorted(zip(area_array, cnts), key=lambda x: x[0], reverse=True)

    c = sorteddata[0][1]
    # M = cv2.moments(c)
    # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    rect = cv2.minAreaRect(c)

    binary_mask = np.zeros((depth_img.size[1], depth_img.size[0], 1), np.uint8)
    cv2.drawContours(binary_mask, [c], -1, (255, 255, 255), -1)
    
    return binary_mask, rect

def get_center_from_rect(rect):
    box = cv2.boxPoints(rect)
    center_x = (box[0][0] + box[2][0]) / 2.0
    center_y = (box[0][1] + box[2][1]) / 2.0
    return center_x, center_y


import torch
import torch.nn as nn
from torch.autograd import Variable


# Depth prediction loss. Code adapted from: https://github.com/fangchangma/sparse-to-dense.pytorch
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss





import torch
import math
import numpy as np


"""
RMSE: rmse
REL: absrel
Delta1: delta1
Delta2: delta2
Delta3: delta3
"""
class DepthEvalResult(object):
    def __init__(self):
        self.rmse = 0
        self.absrel = 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.center_success = 0
        self.iou_score = 0
        self.set_to_worst()
        self.if_using_average_box = False

    def set_to_worst(self):
        self.rmse = np.inf
        self.absrel = np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.center_success = 0
        self.iou_score = 0.

    def evaluate(self, output, target):
        valid_mask = target>0
        output_tmp = output[valid_mask]
        target_tmp = target[valid_mask]

        abs_diff = (output_tmp - target_tmp).abs()

        mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(mse)
        self.absrel = float((abs_diff / target_tmp).mean())

        maxRatio = torch.max(output_tmp / target_tmp, target_tmp / output_tmp)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())

        if type(output) == torch.Tensor:
            output = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
        total_num = output.shape[0]
        success_count = 0.0
        iou_score_lst = []
        for idx in range(total_num):
            pred_mask, pred_rect = get_object_rect_from_depth(output[idx])
            gt_mask, gt_rect = get_object_rect_from_depth(target[idx])
            iou_score = get_iou_score_from_masks(pred_mask, gt_mask)
            iou_score_lst.append(iou_score)
            pred_x, pred_y = get_center_from_rect(pred_rect)
            gt_x, gt_y = get_center_from_rect(gt_rect)
            if self.if_using_average_box:
                dist = np.linalg.norm(np.array((69.72152870269049, 66.0700346719651)) - np.array((gt_x, gt_y))) # cube
            else:
                dist = np.linalg.norm(np.array((pred_x, pred_y)) - np.array((gt_x, gt_y)))
            cross = np.linalg.norm(np.array(gt_rect[1]) - np.array([0, 0]))
            # gt_max = max(gt_rect[1])
            if dist <= cross / 2.0:
                success_count = success_count + 1.0
        self.center_success = success_count
        iou_score_lst = np.array(iou_score_lst)
        self.iou_score = np.mean(iou_score_lst)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # im = cv2.drawContours(im,[box],0,(0,0,255),2)
