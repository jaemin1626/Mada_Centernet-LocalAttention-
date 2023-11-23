from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
import torch.nn.functional as F
from misc import pytorch_ssim

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    if opt.arch == 'kbdg':  
      self.crit = torch.nn.MSELoss(reduction='sum')
    else:
      self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
      self.loss_2 = pytorch_ssim.SSIM(window_size=11)
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss = 0
    loss2 = 0
    if opt.arch == 'kbdg':
      loss = self.crit(outputs, batch)
      loss += 10*self.cos_loss(outputs, batch)
      loss /= self.opt.batch_size
      loss_stats = {'loss': loss, 'hm_loss':loss}
    else:
      batch['hm'] = torch.unsqueeze(batch['hm'], 1)
      
      for s in range(opt.num_stacks):
        if opt.num_stacks == 2:  
          output = outputs[s]
        else:
          output = outputs
        if not opt.mse_loss:
          output['hm'] = _sigmoid(output['hm'])

        if opt.arch == 'sanet':
          hm_loss += self.crit(output, batch['hm'])   
          loss2 = (1 - self.loss_2(output, batch['hm']))
        else:
          hm_loss += self.crit(output, batch['hm']) / opt.num_stacks    
      
      if opt.arch == 'sanet':
        loss = (opt.hm_weight * hm_loss) + (0.001 *loss2)
      else:
        loss = (opt.hm_weight * hm_loss)   
      loss_stats = {'loss': loss, 'hm_loss': hm_loss}
    return loss, loss_stats
  
  def cos_loss(self, output, target):
      B = output.shape[0]
      output = output.view(B, -1)
      target = target.view(B, -1)
      loss = torch.mean(1-F.cosine_similarity(output, target))
      
      return loss
    
class DMTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(DMTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss'] # 수정
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]