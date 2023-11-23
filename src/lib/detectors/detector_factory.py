from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .exdet import ExdetDetector
# from .ddd import DddDetector
from .ctdet import CtdetDetector
from .ctdet_multi import CtdetDetector_multi
# from .ctdet_last_loss import CtdetDetector_last
# from .ctdet_multiStage import CtdetDetector_multiStage
# from .check_densitymap import check
from .DM_multi import DM_multi
# from .multi_pose import MultiPoseDetector
# from .Reg import Regression

detector_factory = {
  # 'exdet': ExdetDetector, 
  # 'ddd': DddDetector,
  'ctdet': CtdetDetector,
  # 'ctdet_last' : CtdetDetector_last,
  'ctdet_multi' : CtdetDetector_multi,
  'densitymap_multi' : DM_multi,
  # 'ctdet_multi_multi_stage' : CtdetDetector_multiStage,
  'densitymap' : DM_multi,
  # 'SANet_multi' : DM_multi,
  # 'multi_pose': MultiPoseDetector,
  # 'reg' : Regression
}
