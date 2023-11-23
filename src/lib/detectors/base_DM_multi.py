from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger


class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    if opt.arch == 'kbdg':
      self.model, _ = create_model(opt.arch, opt.heads, opt.head_conv)
      self.model = load_model(self.model, opt.load_model)
      self.model = self.model.to(opt.device)
    else:
      self.model = create_model(opt.arch, opt.heads, opt.head_conv)
      self.model = load_model(self.model, opt.load_model)
      self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    
    # c = None
    # s = None
    # inp_height, inp_width = self.opt.input_h, self.opt.input_w
    # inp_image = cv2.resize(image, (self.opt.input_h, self.opt.input_w), interpolation=cv2.INTER_AREA)
    # inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
    # images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.opt.down_ratio, # down_ratio 수정 가능
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def run(self, image_or_path_or_tensor, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      images = images.to(self.opt.device)
      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      output, forward_time = self.process(images, return_time=True)
      
      print(output.sum())
      img = output.squeeze(dim=0).detach().cpu().numpy().transpose(1, 2, 0)
      # mean = np.array([[[0.408, 0.447, 0.47]]], dtype = np.float32) 
      # std = np.array([[[0.289, 0.274, 0.278]]], dtype = np.float32)
      # img = ((img * std + mean) * 255).astype(np.uint8)
      
      plt.figure()
      plt.imshow(img, cmap=CM.jet)
      plt.axis('off'), plt.xticks([]), plt.yticks([])
      # ax = plt.gca()
      # ax.axes.xaxis.set_visible(False)
      # ax.axes.yaxis.set_visible(False)
      plt.tight_layout()  
      plt.ioff()
      # plt.cla()
      plt.savefig('../result_image/sanet/' + image_or_path_or_tensor.split("/")[-1], dpi=100, bbox_inches='tight', pad_inches=0)
      
      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time
    
    return {'results': output, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}