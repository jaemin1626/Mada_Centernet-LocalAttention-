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
import math

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

    # print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density_map = np.zeros(img_shape, dtype=np.float32)
    h, w = density_map.shape[:2]
     
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    if adaptive_mode == True: # adaptive_mode == False
        fixed_values = None
        leafsize = 2048
        # tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        # distances, locations = tree.query(points, k=4)
    for idx, (p, radius) in enumerate(zip(points, radius)):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, p[0]), min(w-1, p[1])
        if num_gt > 1:
            if adaptive_mode == 1:
                # sigma = int(np.sum(distances[idx][1:4]) * 0.1)
                pass
            elif adaptive_mode == 0:
                #sigma = fixed_value
                sigma = radius
        else:
            sigma = radius
        sigma = max(1, sigma)
        
        gaussian_radius_no_detection = sigma * 3
        gaussian_radius = gaussian_radius_no_detection

        if fixed_values is not None:
            grid_x, grid_y = int(p[0]//(h/3)), int(p[1]//(w/3))
            grid_idx = grid_x * 3 + grid_y
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

    # print(np.sum(density_map))
    
    if obj_nums == 1:
      density_map = density_map / (np.sum(density_map / 1))
    else:
      density_map = density_map / (np.sum(density_map / num_gt))
    # print(np.sum(density_map))
    return density_map

class KBDG_MultiScale(data.Dataset):
  def _coco_box_to_bbox(self, box): # x, y, w, h
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], # (x, y, x + w, y + h)
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
    c= np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    c_hr = np.array([img.shape[1], img.shape[0]], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        
        # hr_w_border = self._get_border(256, img.shape[1])
        # hr_h_border = self._get_border(256, img.shape[0])
        # c_hr[0] = np.random.randint(low=hr_w_border, high=img.shape[1] - hr_w_border)
        # c_hr[1] = np.random.randint(low=hr_h_border, high=img.shape[0] - hr_h_border)
      else: # 랜덤크롭을 제외하기에 밑에 코드 실행
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

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    hr_output_h = output_h * 4 # 수정
    hr_output_w = output_w * 4 # 수정
    
    origin_h = input_h
    origin_w = input_w
    
    num_classes = self.num_classes

    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
    
    ## HR_heatmap 생성을 위한 아핀 트랜스폼 함수 정의
    trans_output_hr = get_affine_transform(c, s, 0, [hr_output_h, hr_output_w])


    hr_points = [] # 추가

    gt_det = []
    hr_gt_det = []
    for k in range(num_objs):
      ann = anns[k]

      # hr_heatmap 생성하기 위한 bbox 조정
      bbox_hr = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox_hr[[0, 2]] = width - bbox_hr[[2, 0]] - 1
      bbox_hr[:2] = affine_transform(bbox_hr[:2], trans_output_hr)
      bbox_hr[2:] = affine_transform(bbox_hr[2:], trans_output_hr)
      bbox_hr[[0, 2]] = np.clip(bbox_hr[[0, 2]], 0, hr_output_w - 1)
      bbox_hr[[1, 3]] = np.clip(bbox_hr[[1, 3]], 0, hr_output_h - 1)
      hr_h, hr_w = bbox_hr[3] - bbox_hr[1], bbox_hr[2] - bbox_hr[0]
      
      if hr_h > 0 and hr_w > 0:
        #############################################################
        hr_radius = gaussian_radius((math.ceil(hr_h), math.ceil(hr_w))) # object size adaptive standard
        hr_radius = max(0, int(hr_radius))
        
        hr_ct = np.array( # center point calculate
          [(bbox_hr[0] + bbox_hr[2]) / 2, (bbox_hr[1] + bbox_hr[3]) / 2], dtype=np.float32)
        hr_ct_int = hr_ct.astype(np.int32)
        #############################################################
        #############################################################

        hr_points.append(hr_ct_int)
        
        hr_gt_det.append([hr_ct[0] - hr_w / 2, hr_ct[1] - hr_h / 2,
                       hr_ct[0] + hr_w / 2, hr_ct[1] + hr_h / 2, 1, cls_id])

    ret = {'input': inp, 'points' : hr_points, 'file_name' : file_name}
    
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'hr_gt_det':hr_gt_det,'img_id': img_id}
      ret['meta'] = meta
    return ret