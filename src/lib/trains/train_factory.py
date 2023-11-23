from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ctdet_multiScale import CtdetTrainer_MultiScale
# from .ctdet_last_loss import Ctdet_lastloss_Trainer
# from .ctdet_multiScale_last import CtdetTrainer_MultiScale_last
# from .ctdet_multiScale_multiStage import CtdetTrainer_MultiScale_MultiStage
# from .dm import DMTrainer
from .DM_multiScale import DM_MultiScale
from .KDMG_multiScale import KDMG_MultiScale
from .ctdet_multiScale_KDMG import CtdetTrainer_MultiScale_KDMG
from .dm import DMTrainer
# from .ddd import DddTrainer
# from .exdet import ExdetTrainer
# from .multi_pose import MultiPoseTrainer
# from .regression import RegTrainer

# opt.task
train_factory = {
  # 'exdet': ExdetTrainer, 
  # 'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'densitymap' : DMTrainer,
  # 'ctdet_last' : Ctdet_lastloss_Trainer,
  'densitymap_multi' : DM_MultiScale,
  'kbmg_multi' : KDMG_MultiScale,
  # 'SANet_multi' : DM_MultiScale,
  # 'ctdet_multi_last' : CtdetTrainer_MultiScale_last,
  # 'ctdet_multi_multi_stage' : CtdetTrainer_MultiScale_MultiStage,
  'ctdet_multi' : CtdetTrainer_MultiScale,
  'ctdet_multi_kdmg' : CtdetTrainer_MultiScale_KDMG,
  # 'multi_pose': MultiPoseTrainer, 
  # 'reg' : RegTrainer
}
