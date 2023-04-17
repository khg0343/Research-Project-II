### [Research-Project-II] 
# Weakly Supervised Learning for Object Detection
## 연구 목적 (Problem statement)

<aside>
ℹ️ 연구의 주제를 서술하고 그 중요성과 필요성을 설명합니다.

</aside>

본 연구에서는 Object Detection이라는 Task에 Weakly Supervised Learning을 이용하여 모델을 학습시킴으로써 기존 Object Detection이 가지고 있는 비용적 한계를 극복하고, 더 적은 비용으로 기존의 모델과 유사한 성능의 예측도를 가지게 하는 모델을 제안하고자 합니다.

## 연구 배경 (Motivation and background)

<aside>
ℹ️ 연구의 배경을 서술하여 계획하는 연구가 가지는 맥락을 밝힙니다. 기존의 연구의 한계와 문제점을 명확하게 서술합니다.

</aside>

 Computer Vision 분야는 크게 Localization과 Object Detection, 그리고 Segmentation으로 구분됩니다. 그중 Object Detection은 한 개 이상의 Object가 이미지에 존재할 때, 각각의 Object의 위치를 Bounding Box로 지정하는 동시에 Object의 Class를 분류하는 Task를 말합니다. 이때 학습을 위해 대량의 이미지와 그 이미지에 대한 Class에 대한 정보, 그리고 Bounding Box의 좌표가 그 학습 데이터로 들어가게 됩니다. 하지만, 각각의 이미지에 Object를 Labeling을 하고 Bounding Box를 생성하는 과정에서 많은 인력과 시간, 비용이 소모됩니다. 이러한 비용적 한계를 줄이고자 나타난 개념이 Weakly Supervised Learning입니다. 이는 기존의 학습 데이터 Label보다 부족한(Weakly) 정보만을 가지고, 더 많은 결과를 생성하게 합니다. 즉, 이를 Object Detection Task에 적용할 경우, 이미지 Class에 대한 정보(Weakly한 정보)만을 학습 데이터로 입력하여 Classification 뿐만 아니라 Bounding Box까지 탐지하게 하는 것입니다. 이와 관련된 연구 분야를 Weakly Supervised Object Detection (이하 WSOD)이라고 합니다.

 물론 WSOD model의 결과는 비교적 Fully annotated training data로 학습한 모델의 결과보다는 정확도 측면에서 조금 떨어질 수밖에 없습니다. 하지만 학습 데이터를 Labeling하는 작업의 비용적 측면에서의 한계를 극복할 수 있기에, 정확도와 비용 trade-off 사이의 균형을 찾아 효과적인 결과를 내는 것이 이 연구의 가장 큰 의의입니다.

 현재 WSOD분야의 the State-of-the-Art Model은 2020년 NVIDIA와 University of Illinois, University of California에서 함께 제작한 wetectron이라는 Model로, PASCAL VOC과 COCO dataset을 사용한 model중 가장 높은 성과를 냈습니다. 이는 Instance Ambiguity (Object 누락 및 그룹화 등의 결과가 모호하게 나타나는 현상), Part Domination (Object의 일부만 Bounding Box로 처리되는 현상), Memory Consumption과 같은 WSOD에서 흔히 나타나는 문제에 집중하여 연구한 모델입니다.

## 연구 방법 (Research proposal)

<aside>
ℹ️ 문제 해결을 위한 연구 수행 방법을 간략히 서술합니다.

</aside>

 기존에 연구된 자료를 바탕으로 WSOD 모델을 설계 및 재현하고, 다양한 variation을 고려한 새로운 model을 제안해보고자 합니다.

 먼저 WSOD 를 제작하는 데에 필요한 배경 지식을 학습할 계획입니다. 이후, wetectron의 원리와 연구 과정이 기재 되어있는 논문[2]을 바탕으로 그 학습 과정을 이해하고 해당 WSOD Model 를 reproduce 하는 것을 첫 번째 목표로 잡았습니다. 이후 첫 번째 목표의 모델의 개발 진행 경과에 따라, 아래의 세 가지 Variation 들을 고려한 추가적인 연구를 진행해 볼 계획입니다.

1. Transformer for Weakly Supervised Object Detection
현재 많은 분야의 SOTA 급 모델에서 Vision Transformer 를 활용하는 반면, Weakly Supervised Object Detection 의 SOTA 모델에서는 Transformer 를 사용하지 않는 것으로 보였습니다. 이에, Weakly Supervised Object Detection 의 Model 에 Transformer 구조를 적용하는 것에 대한 연구를 생각해 보았습니다.
2. Adversarial Learning for Weakly Supervised Object Detection
모델을 더욱 robust 하게 하기 위한 regularization 방법 중 Adversarial Learnin 을 활용한 방법이 있습니다. ACoL 이라는 model 에서 object localization 분야에서 Adversarial Erasing을 이용한 end-to-end model 이 연구된 바 있습니다. 이에 WSOD 에도 Adversarial Learning를 적용한 model 설계에 대한 연구를 생각해 보았습니다.
3. wetectron model with ImageNet dataset
Weakly Supervised Object Detection 의 SOTA 모델인 wetectron 는 PASCAL VOC 와 COCO 라는 Dataset 을 사용하여 결과를 도출해냈습니다. 하지만, 많은 관련 연구에서 Large Scale 의 dataset 인 ILSVRC(ImageNet)을 사용하는 경우가 많았기에, 해당 SOTA model 에 ImageNet 을 학습한 결과를 확인해보고, 가능하다면 정확도를 올려보는 연구에 대해 생각해보았습니다.

## 기대 효과 (Expected output)

<aside>
ℹ️ 연구 결과를 통해 기대할 수 있는 학계 또는 산업계의 파급 효과를 간략히 서술합니다.

</aside>

 Weakly Supervised Learning 을 통해 Object Detection 라는 Task 를 수행함으로써 적은 시간과 비용의 작업으로 비교적 높은 정확도의 결과를 내어 생산성 측면에서 그 파급효과를 기대할 수 있을 것입니다. 또한, 이미지뿐만 아니라 영상 데이터를 통한 WSOD 는 더욱 다양한 분야에서 활용될 수 있을 것입니다.

학계에서는 WSOD 의 연구 결과를 통해 아래와 같은 분야에서 이용될 수 있습니다.

1. Medical Imaging
Medical Imaging 분야의 경우, Lesion 찾는 문제를 WSOD 를 통해 해결하기도 합니다. Lesion 의 위치에 대한 정보를 알 수 없거나, Lesion 에 대한 single example 이 없을 경우 일반적인 Supervised Learning 으로는 해결하기 어려운데, 이를 Weakly Supervised Learning 을 사용하여 해결할 수 있습니다. 실제로 뇌의 Lesion 의 위치를 나타내는 Attention map 을 WSOD 를 활용하여 계산하는 방법이 연구[4]된 바 있습니다.
2. Remote Sensing
Remote Sensing 분야의 경우, aerial image 에서의 Object Detection 이 중요한 문제로 다뤄져 왔습니다. 하지만 모든 이미지에 labeling 을 하는 것은 전문가의 지식과 많은 시간을 요하는 작업이기에, WSOD 를 활용한 방법을 통해 효과적으로 이러한 문제를 해결할 수 있습니다. 실제로 coarse-grained metadata 만을 학습 데이터로 사용한 model 에 대한 연구[5]가 진행된 바 있습니다.

뿐만 아니라 산업계에서도 불량품 탐지를 통해 제품 생산성을 높이거나, video WSOD를 사용하여 CCTV 영상에서의 이상행동 탐지 및 사고 현장의 사람 탐지하는 등 WSOD는 다양한 분야에서 활용될 수 있습니다.

## 연구 추진 일정 (Future plan)

02/24 지도교수님과 첫 번째 미팅 및 연구 주제 선정
03/17 연구 제안서 (Proposal) 제출
03/20 - 03/31 WSOD 관련 배경 지식 학습 및 개발 환경 구축
04/03 - 04/28 WSOD Model 를 reproduce 를 목표
04/28 연구 진행 보고서 및 비디오 제출
05/01 - 05/24 연구진행상황에 따라 추후에 결정
05/26 최종 포스터 발표 심사 (대면, 심사)
06/02 최종 보고서 제출

## 참고 문헌(References)

[1]Bilen, H. and Vedaldi, A., “Weakly Supervised Deep Detection Networks”, *arXiv eprints*, 2015. doi:10.48550/arXiv.1511.02853.

[2]Ren, Z., “Instance-aware, Context-focused, and Memory-efficient Weakly Supervised Object Detection”, *arXiv e-prints*, 2020. doi:10.48550/arXiv.2004.04725.

[[2]Ren, Z., “Instance-aware, Context-focused, and Memory-efficient Weakly Supervised Object Detection”, <i>arXiv e-prints</i>, 2020. doi:10.48550/arXiv.2004.04725.](https://www.notion.so/2-Ren-Z-Instance-aware-Context-focused-and-Memory-efficient-Weakly-Supervised-Object-Detection-9159260f54ce409686d68e884fb7b4f6)

[3]Zhang, D., Han, J., Cheng, G., and Yang, M.-H., “Weakly Supervised Object Localization and 
Detection: A Survey”, *arXiv e-prints*, 2021. doi:10.48550/arXiv.2104.07918.

[4]Dubost, F., “Weakly Supervised Object Detection with 2D and 3D Regression Neural 
Networks”, *arXiv e-prints*, 2019. doi:10.48550/arXiv.1906.01891.

[5]C. Fasana, S. Pasini, F. Milani, and P. Fraternali, “Weakly Supervised Object Detection for 
Remote Sensing Images: A Survey,” Remote Sensing, vol. 14, no. 21, p. 5362, Oct. 2022, doi: 
10.3390/rs14215362
