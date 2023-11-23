from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import scipy
from scipy.spatial import KDTree
from scipy.ndimage import filters
from numpy import inf
import math

#partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(points, radius, img_size= None, adaptive_mode=False, fixed_value=15, fixed_values=None):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

    return:
    density: the density-map we want. Same shape as input image but only has one channel.

    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape = img_size
      
    obj_nums = len(points)

    print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density_map = np.zeros(img_shape, dtype=np.float32)
    h, w = density_map.shape[:2]
     
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    if adaptive_mode == True:
        fixed_values = None
        leafsize = 2048
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances, locations = tree.query(points, k=4)
    for idx, (p, radius) in enumerate(zip(points, radius)):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, p[1]), min(w-1, p[0])
        if num_gt > 1:
            if adaptive_mode == 1:
                sigma = int(np.sum(distances[idx][1:4]) * 0.1)
            elif adaptive_mode == 0:
                #sigma = fixed_value
                sigma = radius
        else:
            sigma = radius
        sigma = max(1, sigma)
        
        gaussian_radius_no_detection = sigma * 3
        gaussian_radius = gaussian_radius_no_detection

        if fixed_values is not None:
            grid_y, grid_x = int(p[0]//(h/3)), int(p[1]//(w/3))
            grid_idx = grid_y * 3 + grid_x
            gaussian_radius = fixed_values[grid_idx] if fixed_values[grid_idx] else gaussian_radius_no_detection
                    
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        # gaussian_map[gaussian_map < 0.0003] = 0 2022.01.12 주석처리.
        if np.sum(gaussian_map):
            gaussian_map = gaussian_map / np.sum(gaussian_map)
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
    
        density_map[
            max(0, p[0]-gaussian_radius):min(density_map.shape[0], p[0]+gaussian_radius+1),
                max(0, p[1]-gaussian_radius):min(density_map.shape[1], p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]

    print(np.sum(density_map))
    
    if obj_nums == 1:
      density_map = density_map / (np.sum(density_map / 1))
    else:
      density_map = density_map / (np.sum(density_map / num_gt))
    print(np.sum(density_map))
    return density_map

class check_GT(data.Dataset):
  def _coco_box_to_bbox(self, box, keep_resolution): # x, y, w, h
    if keep_resolution:
    # 2022.01.06 밀도맵을 원본 이미지와 동일한 크기로 만듬
      bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], # (x, y, x + w, y + h)
                    dtype=np.float32)
    else:
    # 2022.01.06 밀도맵을 512 x 512로 만듬
      y_ = self.img_height
      x_ = self.img_width
      targetSize = self.output_h
      x_scale = targetSize / x_
      y_scale = targetSize / y_
      (origLeft, origTop, origRight, origBottom) = (box[0], box[1], box[2] + box[0], box[3] + box[1])
      x_min = int(np.round(origLeft * x_scale))
      y_min = int(np.round(origTop * y_scale))
      x_max = int(np.round(origRight * x_scale))
      y_max = int(np.round(origBottom * y_scale))
      bbox = np.array([x_min, y_min, x_max, y_max],
                      dtype=np.float32)

    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    self.img_height, self.img_width = img.shape[0], img.shape[1] # 추가
    c = np.array([img.shape[1] / 2., img.shape[0]/ 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, input_w) # 수정
        h_border = self._get_border(128, input_h) # 수정
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h # 수정 
    output_w = input_w # 수정
    self.output_h = input_h
    self.output_w = input_w
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    
    draw_gaussian = draw_msra_gaussian 
    keep_resolution = False # 2022.01.11 이미지 사이즈와 똑같은 밀도맵을 생성하려면 True로 설정
    
    points = [] # 추가
    radius_list = []
    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'], keep_resolution) # return (x, y, x + w, y + h)
      cls_id = int(self.cat_ids[ann['category_id']])

      # bbox[:2] = affine_transform(bbox[:2], trans_output)
      # bbox[2:] = affine_transform(bbox[2:], trans_output)
      # bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      # bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w))) # object size adaptive standard
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array( # center point calculate
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        
        points.append(ct_int)
        radius_list.append(radius)
    #density = get_density_map_gaussian(points)
    print("pest num", len(points))
    
    # 원본 이미지 사이즈에 맞춰 덴시티맵을 생성하기 위해선 이미지의 높이, 너비가 필요
    # 512 x 512 덴시티 맵을 생성하고 싶으면 아래의 img_size 주석넣으면 된다.
    if keep_resolution:
      img_size = (img.shape[0], img.shape[1])
    else:
      img_size = (512, 512)
    
    density = gaussian_filter_density( points, radius_list, img_size = img_size, adaptive_mode=False)

    ret = {'input': inp, 'hm': hm, 'density':density, 'file_name' : file_name}
    
    return ret