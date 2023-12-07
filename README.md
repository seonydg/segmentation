# segmentation

1. u-net
1.1 u-net
1.2 u-net transfer
2. Mask R-CNN
3. DeepLabV3Plus

   
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


---
# 2. Mask R-CNN
# Mask R-CNN(ICCV 2017)

R-CNN의 마지막 논문으로, 기존의 classification, object detection과는 다른 task를 수행한다. Semantic Segmentation은 클래스별로 영역을 필셀별로 구분하는데, 같은 클래스는 같은 영역으로 되어 있는 반면, Instance Segmentation은 클래스별로도, 즉 하나의 객체당 Segmentation를 구별을 한다.
바운딩 박스 정보를 같이 이용하면 클래스별로 구별하여 Instance Segmentation을 수행하게 된다.

![](https://velog.velcdn.com/images/seonydg/post/68101410-f579-4a9a-8764-16a563942efa/image.png)


구현은 간단하게 진행이 되는데, Faster R-CNN을 backbone을 활용하여 7×7 레졸루션을 늘이고 채널을 줄여나간다. 마지막 레이어의 80은 coco dataset의 클래스 수를 의미한다.
3가지 class, box, mask를 수행한다.

![](https://velog.velcdn.com/images/seonydg/post/820b74d2-4c44-4c82-90fe-95201643ac59/image.png)


ROIPooling의 경우 기존의 feature map에 있는 값을 그대로 가져온 반면, ROIAlign은 feature map의 필셀별로 딱 맞는 것이 아닌 소수점도 허용한다. 해당 ROI window에 있는 필셀의 위치 좌표를 이용해서 주변에 있는 feature map의 픽셀 값들도 참조해서 값을 정한다.

![](https://velog.velcdn.com/images/seonydg/post/0e57187e-7316-4624-afd3-88f2fb77c75d/image.png)


결과들을 보면, 성능이 좋다.


![](https://velog.velcdn.com/images/seonydg/post/b2b80c8a-fca3-4aba-83fd-8b9b6706f6d3/image.png)


![](https://velog.velcdn.com/images/seonydg/post/af6e1add-bece-483b-ae12-08d57e79f243/image.png)


softmax와 sigmoid를 사용했을 때, 바운딩 박스의 탐지가 더 올라가는 점은 흥미롭다.

![](https://velog.velcdn.com/images/seonydg/post/ea5b7179-aee8-4dd7-8e89-1bf3cfdf547b/image.png)


object detection에서 Faster R-CNN에 RoIAlign을 적용하면 Mask R-CNN과 아키텍쳐는 같지만, Mask R-CNN의 성능이 더 좋은 이유는 multi-task learning을 진행하면 성능이 더 올라간다고 논문에서는 주장하고 있다.

![](https://velog.velcdn.com/images/seonydg/post/c2e5c3b0-8051-4be5-9c5b-6c02098b686a/image.png)

---
# 3. DeepLabV3Plus
# DeepLabV3Plus(ECCV 2018)

아래의 그림과 같이 사람과 background를 구분하는 것처럼, 입력 이미지에서 의미를 갖는 영역을 필셀별로 구분해내는 작업을 Semantic Segmentation이라고 한다.

![](https://velog.velcdn.com/images/seonydg/post/20eaa5b1-9058-4640-b667-008f2400b4ff/image.png)

DeepLab versions
- DeepLab V1 : ICLR 2105
- DeepLab V2 : TPAMI 2017
- DeepLab v3 : ECCV 2018

DeepLabV3Plus는 현재까지 준수한 성능을 가지고 있다. 그리고 모바일 버전의 디바이스에서도 real time으로 작동을 한다.

![](https://velog.velcdn.com/images/seonydg/post/f826cd5f-7736-40e3-a77f-35de98c0b1df/image.png)


DeepLabV3Plus는 3가지 architecture improments를 가지고 있다.

기존에 모든 입력 체널에 필터를 연산한 것과는 다르게, **Depthwise conv**는 하나의 채널당 다른 필터를 적용하여 연산을 한다. 그리고 구멍을 뚫는 식으로 격자 무늬를 넓혀가면서 conv 연산을 진행하는 것을 제안한다.

![](https://velog.velcdn.com/images/seonydg/post/72c739c4-ecf1-4d47-87c0-9d34ca3176b4/image.png)

레졸루션을 줄여가며 컨볼루션 리셉티브 필드가 다른 풀링 레이어들(spatial pyramid pooling)을 거쳐서 가장 작은 레졸루션을 가진 feature map을 생성하고 업샘플리을 해서 에측을 진행한 것과 U-Net처럼 레졸루션을 줄이고 키우는 과정을 합치는 방법을 제안한다.
레졸루션을 4배로 키우고 다시 4배로 최종 16배로 키우는데, 중간 단계에서 연산을 더 하지 않는 이유는 성능 뿐만 아니라 효율도 좋게 만들기 위해서다.

![](https://velog.velcdn.com/images/seonydg/post/2fdd4e52-fb11-49a5-8d85-936f327280e0/image.png)

![](https://velog.velcdn.com/images/seonydg/post/9bb12be1-bbad-41e8-bb14-0334d4fb2b2a/image.png)

