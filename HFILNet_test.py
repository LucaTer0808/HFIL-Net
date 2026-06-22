# -*- coding: utf-8 -*-
"""
@author: gaohaoran@Dalian Minzu University
@software: PyCharm
@file: HFIL-Net.py
@time: 2023/11/23 7:23
"""

import torch
import torch.nn.functional as F
import sys
import shutil
import time

sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.HFILNet import SwinTransformer, HFILNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
# NEU: Die drei Quellpfade als separate, verpflichtende Argumente
parser.add_argument('--rgb_path', type=str, required=True, help='path to RGB images')
parser.add_argument('--gt_path', type=str, required=True, help='path to Ground Truth (GT) images')
parser.add_argument('--depth_path', type=str, required=True, help='path to depth maps')
parser.add_argument('--target', type=str, required=True, help='the directory to save the test results')
opt = parser.parse_args()

# set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

# load the model
model = HFILNet()
model.load_state_dict(torch.load('caps/HFILNet_epoch_best.pth'))
model.cuda()
model.eval()

# Zielordner vorbereiten (Idempotent)
save_path = opt.target + "/"
if os.path.exists(save_path):
    print(f"{save_path} already exists, deleting it")
    shutil.rmtree(save_path)
os.makedirs(save_path, exist_ok=False)

# Slashes am Ende garantieren, falls sie beim Aufruf vergessen wurden
image_root = opt.rgb_path if opt.rgb_path.endswith('/') else opt.rgb_path + '/'
gt_root = opt.gt_path if opt.gt_path.endswith('/') else opt.gt_path + '/'
depth_root = opt.depth_path if opt.depth_path.endswith('/') else opt.depth_path + '/'

# Test-Loader direkt mit den übergebenen Argumenten füttern
test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)

start_time = time.time()

for i in range(test_loader.size):
    image, gt, depth, name, image_for_post = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    
    depth = depth.repeat(1, 3, 1, 1).cuda()
    out1, out2, out3 = model(image, depth)

    out1 = F.upsample(out1, size=gt.shape, mode='bilinear', align_corners=False)
    out2 = F.upsample(out2, size=gt.shape, mode='bilinear', align_corners=False)
    out3 = F.upsample(out3, size=gt.shape, mode='bilinear', align_corners=False)

    out1 = out1.sigmoid().data.cpu().numpy().squeeze()
    out2 = out2.sigmoid().data.cpu().numpy().squeeze()
    out3 = out3.sigmoid().data.cpu().numpy().squeeze()

    out1 = (out1 - out1.min()) / (out1.max() - out1.min() + 1e-8)
    out2 = (out2 - out2.min()) / (out2.max() - out2.min() + 1e-8)
    out3 = (out3 - out3.min()) / (out3.max() - out3.min() + 1e-8)
    print('save img to: ', save_path + name)

    os.makedirs(os.path.dirname(save_path + name), exist_ok=True)
    cv2.imwrite(save_path + name, out3 * 255)

end_time = time.time()
total_time = end_time - start_time
time_per_image = total_time / test_loader.size
print(f'Total testing time: {total_time:.2f} seconds')
print(f'Average time per image: {time_per_image:.2f} seconds')

eval_path = os.path.join(save_path, 'evaluate.txt')
with open(eval_path, 'w') as f:
    f.write(f'Total images tested: {test_loader.size}\n')
    f.write(f'Total testing time: {total_time:.2f} seconds\n')
    f.write(f'Average time per image: {time_per_image:.2f} seconds\n')

print('Test Done!')

# #######################################################  end  ######################################