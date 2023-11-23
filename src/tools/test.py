from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import stat

OUT_PATH = '../../models/reg_pest_hg.pth'

import torch
state_dict = torch.load(OUT_PATH)

for i,j in state_dict.items():
    print(i,j)