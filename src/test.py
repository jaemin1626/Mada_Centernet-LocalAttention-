from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import pycocotools.coco as coco
import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import pandas as pd


from lib.opts_custom import opts # Centernet

from lib.logger import Logger
from lib.utils.utils import AverageMeter
from lib.datasets.dataset_factory import dataset_factory
from lib.detectors.detector_factory import detector_factory
from lib.datasets.dataset_factory import get_dataset
from lib.opts_madaCenternet_localAttention import opts # 수정

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.data_dir = os.path.join(opt.data_dir, 'coco') # 수정    
    self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_{}2017.json').format('val')
    
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)    
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
    self.max_objs = 250
    #self.label_path = dataset.label_dir #추가

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    
    # 추정 해충 갯수 구하기 위한 정보 가져오기
    # label = pd.read_csv(self.label_path, header=None, index_col = None)
    file_name = img_info['file_name'].split('.')[0]
    # label = label[label[0] == file_name]
    # label_num = int(label[1])

    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id,  num_objs, file_name, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  confidence_thr = 0.3 # 추가
  mae = [] # 추가
  rmse = [] # 추가
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  
  save_label_num = {} # 추가
  file_name_list = [] # 추가
  #path = "/home/ai001/MadaCenternet_(LocalAttention)/result/groundtruth/"
  
  for ind, (img_id, num_objs, file_name, pre_processed_images) in enumerate(data_loader):
    # preidct
    ret = detector.run(pre_processed_images,file_name = file_name)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    
    # 추가
    file_name_list.append(file_name)
    results2 = ret['results']
    count_bbox = 0
    for j in range(1, 2):
      for bbox in results2[j]:
        if bbox[4] > confidence_thr:
          count_bbox += 1
    data = pre_processed_images['image'][0]
    name = file_name[0]
    num  = str(int(num_objs[0]))
    #cv2.imwrite(path + "{}_{}.png".format(num,file_name[0]), data.numpy())
    
    save_label_num[img_id] = [num_objs.item(), count_bbox] # 추가
    #cv2.imwrite(path + "/pred" +'/pred_{}_{}.png'.format(count,image_path[0]), v)
    mae.append(np.abs(num_objs - count_bbox)) # 추가
    rmse.append((num_objs - count_bbox)**2) # 추가 
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()

  mae = torch.mean(torch.Tensor(mae)) # 추가
  rmse = torch.sqrt(torch.mean(torch.Tensor(rmse))) # 추가

  # calculate MAE
  print('Print MAE :', mae.item())
  print('Print RMSE :', rmse.item())
  
  # calculate mAP
  dataset.run_eval(results, opt.save_dir)

  # 원본해충갯수와, 추정해충갯수, 편차를 구하기 위해 추가
  tmp1 = []; tmp2 = []; tmp3 = []; 
  for value in save_label_num.values():
      tmp1.append(value[0])
      tmp2.append(value[1])
      tmp3.append(abs(value[0] - value[1]))

  for i in range(10):
    print(i, ":", tmp3.count(i))

  # 원본영상의 해충 갯수와 추정 해충 개수를 저장하기 위한 csv파일 생성
  df = pd.DataFrame(data=(zip(file_name_list, tmp1, tmp2, tmp3)), columns=['파일이름', '원본해충갯수', '추정해충갯수', '편차'])
  df.to_csv('../exp/coco_hg/label_num.csv', index=False, encoding = 'utf-8-sig')
  
def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
      test(opt)
  else:
      prefetch_test(opt)