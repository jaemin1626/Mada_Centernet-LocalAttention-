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

class CTDetDataset_MultiScale_KDMG(data.Dataset):
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
    hr_output_h = output_h * 2 # 수정
    hr_output_w = output_w * 2 # 수정
    ori_output_h = input_h
    ori_output_w = input_w
    
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
    
    ## HR_heatmap 생성을 위한 아핀 트랜스폼 함수 정의
    trans_output_hr = get_affine_transform(c, s, 0, [hr_output_h, hr_output_w])
    
    trans_ori_output_hr = get_affine_transform(c, s, 0, [ori_output_h, ori_output_w])
    
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    
    hr_hm = np.zeros((num_classes, hr_output_h, hr_output_w), dtype=np.float32) # 추가
    hr_wh = np.zeros((self.max_objs, 2), dtype=np.float32) # 추가
    hr_reg = np.zeros((self.max_objs, 2), dtype=np.float32) # 추가
    hr_ind = np.zeros((self.max_objs), dtype=np.int64) # 추가
    hr_reg_mask = np.zeros((self.max_objs), dtype=np.uint8) # 추가
    
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian # use this

    gt_det = []
    hr_gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox']) # return (x, y, x + w, y + h)
      
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      
      # hr_heatmap 생성하기 위한 bbox 조정
      bbox_hr = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox_hr[[0, 2]] = width - bbox_hr[[2, 0]] - 1
      
      copybbox = bbox_hr.copy()
      bbox_hr[:2] = affine_transform(bbox_hr[:2], trans_output_hr)
      bbox_hr[2:] = affine_transform(bbox_hr[2:], trans_output_hr)
      bbox_hr[[0, 2]] = np.clip(bbox_hr[[0, 2]], 0, hr_output_w - 1)
      bbox_hr[[1, 3]] = np.clip(bbox_hr[[1, 3]], 0, hr_output_h - 1)
      hr_h, hr_w = bbox_hr[3] - bbox_hr[1], bbox_hr[2] - bbox_hr[0]
      
      ori_points = [] # 추가
      # ori_heatmap 생성하기 위한 bbox 조정
      bbox_ori = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(self.cat_ids[ann['category_id']])
      # if copybbox[0]-10 < bbox_hr[0] < copybbox[0]+10 or copybbox[2]-10 < bbox_hr[2] < copybbox[2]+10:
      #   print("asd")
      # if flipped:
      #   bbox_hr[[0, 2]] = width - bbox_ori[[2, 0]] - 1
      # if copybbox[0]-10 < bbox_hr[0] < copybbox[0]+10 or copybbox[2]-10 < bbox_hr[2] < copybbox[2]+10:
      #   print("asd")
      bbox_ori[:2] = affine_transform(bbox_ori[:2], trans_ori_output_hr)
      bbox_ori[2:] = affine_transform(bbox_ori[2:], trans_ori_output_hr)
      bbox_ori[[0, 2]] = np.clip(bbox_ori[[0, 2]], 0, hr_output_w - 1)
      bbox_ori[[1, 3]] = np.clip(bbox_ori[[1, 3]], 0, hr_output_h - 1)
      ori_h, ori_w = bbox_ori[3] - bbox_ori[1], bbox_ori[2] - bbox_ori[0]
      
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w))) # object size adaptive standard
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        #############################################################
        hr_radius = gaussian_radius((math.ceil(hr_h), math.ceil(hr_w))) # object size adaptive standard
        hr_radius = max(0, int(hr_radius))
        ############################################################
        ct = np.array( # center point calculate
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        
        hr_ct = np.array( # center point calculate
          [(bbox_hr[0] + bbox_hr[2]) / 2, (bbox_hr[1] + bbox_hr[3]) / 2], dtype=np.float32)
        hr_ct_int = hr_ct.astype(np.int32)
        #############################################################
        ori_ct = np.array( # center point calculate
          [(bbox_hr[0] + bbox_hr[2]) / 2, (bbox_hr[1] + bbox_hr[3]) / 2], dtype=np.float32)
        
        ori_ct_int = ori_ct.astype(np.int32)
        
        ori_points.append(ori_ct_int)
        #############################################################
        draw_gaussian(hm[cls_id], ct_int, radius) # LR 히트맵에 대한 GT 생성, (부류, 좌표, 가우시안 커널의 분포)
        draw_gaussian(hr_hm[cls_id], hr_ct_int, hr_radius) # HR 히트맵에 대한 GT 생성
        #############################################################
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        #############################################################
        hr_wh[k] = 1. * hr_w, 1. * hr_h
        hr_ind[k] = hr_ct_int[1] * hr_output_w + hr_ct_int[0]
        hr_reg[k] = hr_ct - hr_ct_int
        hr_reg_mask[k] = 1
        
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        hr_gt_det.append([hr_ct[0] - hr_w / 2, hr_ct[1] - hr_h / 2,
                       hr_ct[0] + hr_w / 2, hr_ct[1] + hr_h / 2, 1, cls_id])
        
    ret = {'input': inp, 'points':ori_points, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'hr_reg_mask': hr_reg_mask, 'hr_ind':hr_ind, 'hr_wh':hr_wh, 'file_name' : file_name}
    
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
      ret.update({'hr_reg':hr_reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'hr_gt_det':hr_gt_det,'img_id': img_id}
      ret['meta'] = meta
    return ret