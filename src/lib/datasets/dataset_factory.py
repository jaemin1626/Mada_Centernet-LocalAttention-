from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset
from .sample.make_density_GT import check_GT
from .sample.ctdet_multi_scale import CTDetDataset_MultiScale
from .sample.DM_multi_scale import DMDataset_MultiScale
from .sample.KBDG_multi_scale import KBDG_MultiScale
from .sample.check_DM_multiscale import check_DMDataset_MultiScale
from .sample.DM import DMDataset
from .dataset.coco import COCO
from .dataset.pest_coco import Pest_COCO
from .dataset.pest_coco2 import Pest_COCO2
from .sample.ctdet_multi_scale_KDMG import CTDetDataset_MultiScale_KDMG

# opt.dataset
dataset_factory = {
  'coco': COCO,
  # 'pascal': PascalVOC,
  # 'kitti': KITTI,
  # 'coco_hp': COCOHP,
  'pest_coco' : Pest_COCO,
  'pest_coco_2' : Pest_COCO2,
}
# opt.task
_sample_factory = {
  # 'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'densitymap' : DMDataset,
  'ctdet_multi' : CTDetDataset_MultiScale,
  'densitymap_multi' : DMDataset_MultiScale,
  'kbmg_multi' : KBDG_MultiScale,
  'ctdet_multi_kdmg' : CTDetDataset_MultiScale_KDMG,
  # 'ddd': DddDataset,
  # 'multi_pose': MultiPoseDataset,
  # 'reg' : RegDataset,
  'check_GT' : check_GT
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset

