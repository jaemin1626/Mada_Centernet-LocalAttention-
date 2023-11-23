from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import stat

MODEL_PATH = '../../models/model_for_pest.pth'
OUT_PATH = '../../models/reg_pest_hg.pth'

import torch
state_dict = torch.load(MODEL_PATH)

key_map = ['wh', 'reg']

del state_dict['epoch']
del state_dict['optimizer']

out = {}
for k in state_dict.values():
  for i, j in k.items():
    if key_map[0] in i or key_map[1] in i:
      continue
    else:
      out[i] = j

for i in out:
  print(i)
  
data = {'epoch': 0,
        'state_dict': out}
torch.save(data, OUT_PATH)
