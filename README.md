# segmentation

1. U-Net
1.1 U-Net
1.2 U-Net transfer
2. U-Net++
3. Mask R-CNN
4. DeepLabV3Plus

   
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
# 2. U-Net++
# U-Net++


## U-Net

U-Net은 아키텍쳐 모양이 'U'자형으로 되어 있어 붙여진 이름이다.
기존 CNN은 input image의 레졸루션을 줄여가며 연산을 수행한 반면, U-Net은 레졸루션을 기존처럼 줄여가며 컨볼루션 연산을 수행하고 그 다음에는 2배씩 레졸루션을 키워가며 컨볼루션 연산을 수행한다.
그리고 down sampling때의 같은 레졸루션 크기를 갖는 feature map를 up sampling때의 feature map과 합치는 연산을 수행한다.
하이 레벨의 정보와 로우 레벨 정보를 함께 보존하면서 출력을 학습할 수 있도록 구성이 되어 있다.

![](https://velog.velcdn.com/images/seonydg/post/fb4b2382-35e9-4a32-bbc1-70479a23c223/image.png)


## U-Net++

**U-Net ++**의 원문은 **A Nested U-Net Architecture for Medical Image Segmentation**이다.

잠시 DenseNet(CVPR 2017)을 잠시 보면 아래의 그림과 같이, Input부터 다음 블럭'들'로 skip connection을 각각 넘겨준다. 그 다음 블럭에서 다음 블럭'들'로 다시 skip connection을 각각 넘겨주고 이 전체를 댄스 블럭 하나로, 이런 식으로 댄스 블럭 몇 개를 쌓는 구조로 만들어진다.

![](https://velog.velcdn.com/images/seonydg/post/3aff08c7-9330-42d6-9c5d-ebe4aa189def/image.png)

아래의 그림은 **U-Net ++** 아키텍처다.
검정색으로 된 것이 기존의 U-Net 아키텍처이고, 녹색 Convolution과 다운 샘플링 과정 각각에서 업 샘플링하는 녹색과 파란색의 skip connection이 합쳐지는 Convolution이 추가 되었고 각각의 제일 윗 단에서 Loss가 계산되는 것이 추가가 된 형태로 아키텍처가 구성되어 있다.
그리고 DenseNet과 같이 같은 레졸루션을 가지는 Convolution'들'에 skip-connection을 넘겨준다.
Convolution X에서 앞자리는 다운 샘플링을 하면 순서이고 뒷자리는 업 샘플링의 순서를 말한다.

제일 윗단만 보면 Dense block과 똑같은 형태를 띄고 있는데, 추가적으로 각각의 Convolution에 업 셈플링된 Convolution이 Skip_connection이 추가로 합처진다.
그래서 제일 오른쪽 부분을 제외하더라도 나머지 부분들이 U-Net 아키텍처가 되는 것처럼 구성이 되어 있다.

![](https://velog.velcdn.com/images/seonydg/post/26b634a6-4c03-4e06-9a71-76dbd6d4b34a/image.png)

**U-Net ++** 윗단을 다시 살펴보면, H가 Convolution이고 H(X0, 1)은 H(X0, 0)과 업 샘플링 된 U(X1, 0)이 합쳐지는 식으로 구성되어 있다.

![](https://velog.velcdn.com/images/seonydg/post/e3d83fbf-10eb-4def-84ec-29c007ecefd5/image.png)

해당 논문에서는 이러한 아키텍처의 구성을 잘 살리기 위해서 Deep supervision이라고 하는 Loss를 추가를 한다. 래의 그림과 같이 제일 오른쪽 업 샘플링단을 없애더라도 결과가 도출이 될 수 있도록 각 끝단에 Loss를 추가를 했다.
이렇게 Loss를 구현했을 때 장점이 Network pruning을 inference time에 할 수 있다는 것이다. 그래서 계산량이 많을 때나 적을 때, 즉 가용할 리소스가 많을 때나 적을 때나 예측을 수행할 수 있어야 한다는 것이다.
가용할 리소스가 적다면 일단 **U-Net++L1**에서 결과를 출력하고 그 다음으로 **U-Net++L2**에서 결과를 출력하는 식이다.

![](https://velog.velcdn.com/images/seonydg/post/25b3755b-bf95-469f-a24a-60febd18e2d2/image.png)


**Loss Function**은 binary classification을 수행한다. 메디컬 이미지의 대부분의 데이터셋은 대부분 Yes/No로 구분되어 있다.
Y는 label, Y hat은 모델 output을 말하고, 수식은 binary class에 대한 corss entropy식과 dice coefficient score(산술기하평균)의 합에 대한 N(이미지 수)를 가지고 Loss를 계산한다. 

![](https://velog.velcdn.com/images/seonydg/post/ef7802a3-3742-43a3-bf83-1777de12c9e5/image.png)



## Results

Ground Truth에 대비하여 U-Net ++가 비교적 다른 비교군보다 성능적으로 좋아보이는 결과를 도출한다.

![](https://velog.velcdn.com/images/seonydg/post/54d9ba66-6bf4-4e6c-b961-00fb23c5d824/image.png)

아래는 Segmentation 결과를 IoU로 것인데 좋은 성능을 이끌어낸다고 볼 수 있다.
하지만 DS(Deep Supervision)를 중간 단계에서 쓴 다는 것은, 최종 결과에서 뿐만 아니라 중간에서도 Loss를 계산한다는 것인데, 이것은 오히려 전체 Loss를 계산하는데 방해가 되는 부분임에도 불구하고 더 높은 결과가 도출 되는 것은 눈여겨 볼 만 하다.

![](https://velog.velcdn.com/images/seonydg/post/dee6a54c-3481-4010-84bd-c22853666222/image.png)

아래는 Network pruning에 대한 결과다.

![](https://velog.velcdn.com/images/seonydg/post/fd62dacb-25ad-4afd-8b8e-59d1ee1db14c/image.png)

---
# 3. Mask R-CNN
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
# 4. DeepLabV3Plus
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

