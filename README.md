# segmentation

1. u-net
1.1 u-net
1.2 u-net transfer

   
---
# 1. U-Net
# U-Net(MICCAI 2015)

처음에는 바이오 메디칼 이미지의 Segmentation을 수행한 연구인데,  바이오 메디칼 뿐만이 아니라 다양한 영역에서 U-Net 스타일의 아키텍쳐를 활용이 된다.

U-Net은 아키텍쳐 모양이 'U'자형으로 되어 있어 붙여진 이름이다. 
기존 CNN은 input image의 레졸루션을 줄여가며 연산을 수행한 반면, U-Net은 레졸루션을 기존처럼 줄여가며 컨볼루션 연산을 수행하고 그 다음에는 2배씩 레졸루션을 키워가며 컨볼루션 연산을 수행한다.
그리고 down sampling때의 같은 레졸루션 크기를 갖는 feature map를 up sampling때의 feature map과 합치는 연산을 수행한다.
하이 레벨의 정보와 로우 레벨 정보를 함께 보존하면서 출력을 학습할 수 있도록 구성이 되어 있다.

![image](https://github.com/seonydg/segmentation/assets/85072322/5378896b-0382-41f7-b57e-7a044b43a22f)


논문에서는 feature map 크기를 유지하기 위한 padding을 사용하지 않고 진행한다. 메디칼 영상 데이터 특성상, 데이터의 수가 너무 적다 보니 padding에 의해서 생기는 bias들을 처리하기가 쉽지 않았던 것으로 예측된다.
그래서 conv를 지날 때 마다 feature의 레졸루션이 2픽셀씩 줄어드는 것을 모델 아키텍쳐에서 확인할 수 있다.

![image](https://github.com/seonydg/segmentation/assets/85072322/92cad790-9954-4a9f-a502-fa8cdcb783b1)


up-conv때에는 **1 by 1** 레졸루션의 feature를 **2 by 2** 레졸루션으로 mapping을 하게 된다.

![image](https://github.com/seonydg/segmentation/assets/85072322/20b6274c-503d-4d7e-9304-db1ba023d534)


눈문에서는 메디칼의 특성상 곡선의 형태가 유용하게 사용되기에, Augmentation에 곡선을 추가로 제시한다. 

![image](https://github.com/seonydg/segmentation/assets/85072322/bbc962a4-2eda-4be5-b193-efcaa1c862c1)


그리고 객체들 사이의 구분하기 위한 엣지들을 강조하기 위해 weight mask를 생성하는 방법을 제시한다. 아래의 오른쪽처럼 빨간색은 높은 값을 파란색은 낮은 값을 가지도록 하여, 객체들은 낮은 값을, 백그라운드는 어느 정도의 낮은 값을, 객체들 사이의 엣지들을 높은 값을 가지게 하여 마스크를 생성하는 방법을 제시한다.

![image](https://github.com/seonydg/segmentation/assets/85072322/c951b364-ff60-4a6b-8179-17af4b7db2a8)


Warping Error는 객체들의 분할 및 병합이 잘 되었는지 확인하는 지표로, 객체 사이의 위상의 구조적 의미를 보존하는데 의미를 둔다.

![image](https://github.com/seonydg/segmentation/assets/85072322/4ea61636-68d9-4c36-ab66-cdf091af8ecc)


객체의 모양이 서로 다르더라도 모양에 맞게 잘 탐지를 하는 모습이다. 세포 바깥 부분에 비해서 안 쪽의 부분의 경계가 강하게 표현이 되더라도 잘 구분을 하는 것을 볼 수 있다.


![image](https://github.com/seonydg/segmentation/assets/85072322/5616dc2b-408e-4a7b-a2b2-eeaee718eb6f)

