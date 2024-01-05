## Proposed Pest Object Detection
<li>객체와 배경 영역을 분리하기 위한 객체 및 배경 마스킹 모듈과 이를 기반으로 객체 영역의 특징 구별력을 강화하기 위한 멀티스케일 로컬 어텐션 퓨전 모듈을 제안하고자 한다.</li>

## Proposed Pest Mada-Ceterent(Local Attention) Architecture
<img src="./image/LocalAttention(Mada-Centerent).png"/>

## Trainning Environment
<li> python = 3.8.18 </li>
<li> pytorch = 1.13.0+cu117 </li>
<li> GPU = A100 </li>
<li> CPU = Intel(R) Xeon(R) Gold 5317 CPU @ 3.00GHz </li>
<li> epoch = 100 </li>
<li> batch size = 8 </li>
<li> learning rate = 0.001 </li>
<li> optimizer = Adam </li>

## Evaluation
|      Methods           |    MAE    |   MAP    |
|      -------           |   ----    |   ----   |
|     Retinanet          |  4.634    |  10.449  |
|     Faster RCNN        |  3.312    |  10.297  |
|     RepPoints          |  1.471    |  3.436   |
|     Centernet          |  0.766    |  1.981   |
|     Mada-Centernet     |  0.696    |  1.806   |
|     Proposed Centernet |  0.640    |  1.602   |

## Results
<li> 실험 결과를 통해, 제안한 로컬 어텐션 기반의 MaDa-CenterNet(Local Attention)이 기존의 MaDa-CenterNet의 성능을 개선할 수 있었으며 제안한 로컬 어텐션 모델이 해충 카운팅에 효과적임을 입증하였다. </li>
<img src="./image/Result_image.png"/>
