from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.cm as CM
from progress.bar import Bar
import time
import torch
from lib.models.decode import _nms
try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_DM_multi import BaseDetector

class DM_multi(BaseDetector):
  def __init__(self, opt):
    super(DM_multi, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      if self.opt.arch == 'iccnet':
        output = self.model(images)[0]
      elif  'hourglass' in self.opt.arch:  
        output = self.model(images)[-1]
      else:
        output = self.model(images)
        
      ###########################################################################
      # print(output.sum())
      # # check heat map, apply sigmoid function of heat map, apply NMS of heat map
      # output = output.squeeze(dim=0)
      # img = output.detach().cpu().numpy().transpose(1, 2, 0)
      # # mean = np.array([[[0.408, 0.447, 0.47]]], dtype = np.float32) 
      # # std = np.array([[[0.289, 0.274, 0.278]]], dtype = np.float32)
      # # img = ((img * std + mean) * 255).astype(np.uint8)
      
      # plt.figure(figsize=(2.1,2.1))
      # plt.imshow(img, cmap=CM.jet)
      # plt.axis('off'), plt.xticks([]), plt.yticks([])
      # # ax = plt.gca()
      # # ax.axes.xaxis.set_visible(False)
      # # ax.axes.yaxis.set_visible(False)
      # plt.tight_layout()  
      # plt.ioff()
      # # plt.cla()
      # plt.savefig('../result_image/0.jpg', dpi=100, bbox_inches='tight', pad_inches=0)
    
    
    #####################################
      # hm_sigmoid = hm[0].detach().cpu().numpy().transpose(1, 2, 0)
      # hm_sigmoid = ((hm_sigmoid * std + mean) * 255).astype(np.uint8)
      
      # hm_sigmoid_tmp = hm.contiguous()
      # hm_sigmoid_tmp = _nms(hm_sigmoid_tmp)
      # predict_heat = hm_sigmoid_tmp[0].detach().cpu().numpy().transpose(1, 2, 0)
      # predict_heat = ((predict_heat * std + mean) * 255).astype(np.uint8)

      # img = cv2.resize(img, dsize=(512, 512))
      # hm_sigmoid = cv2.resize(hm_sigmoid, dsize=(512, 512))
      # predict_heat = cv2.resize(predict_heat, dsize=(512, 512))

      # cv2.imwrite('../result_image/heat_map_result/predict_heat.jpg', predict_heat)
      # cv2.imwrite('../result_image/heat_map_result/before_hm_sigmoid.jpg', img)
      # cv2.imwrite('../result_image/heat_map_result/after_hm_sigmoid.jpg', hm_sigmoid)
      ###
      
      torch.cuda.synchronize()
      forward_time = time.time()
      
    if return_time:
      return output, forward_time
    else:
      return output

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)
