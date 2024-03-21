### R-CNN 모델의 동작 순서
##### 1. Selective search 알고리즘을 통해 객체가 있을 법할 위치인 후보 영역(region proposal)을 2000개 추출하여, 각각을 227x227 크기로 warp시킴.
##### 2. Warp된 모든 region proposal을 Fine tune된 AlexNet에 입력하여 2000x4096 크기의 feature vector를 추출.
##### 3. 추출된 feature vector를 linear SVM 모델과 Bounding box regressor 모델에 입력하여 각각 confidence score와 조정된 bounding box 좌표를 얻음. 
##### 4. 마지막으로 Non maximum suppression 알고리즘을 적용하여 최소한의, 최적의 bounding box를 출력.   

### 1. Selective Search
##### Selective Search 알고리즘을 이용해 이미지에서 후보 영역인 Region Proposal 2000개를 추출한다. 추출된 Region Proposals은 AlexNet에 입력하기 위해 227*227크기로 warp시켜준다.   

### 2. Feature Extraction by Fine Tuned AlexNet
##### 추출한 2000개의 Region Proposals을 Fine tune한 AlexNet에 입력해 2000(후보 영역 수) * 4096(Feature Vector의 차원) 크기의 feature vector를 추출한다.   

##### Fine tune시 객체의 수가 N개라고 하면 배경을 포함해 N+1개의 class를 예측하도록 모델을 설계해야하고, 객체와 배경을 모두 포함한 학습 데이터를 구성해야한다. PASCAL 데이터셋에 Selective Search 알고리즘을 적용해 Region Proposals을 추출하고, groun truth box(정확한 bounding box)와의 IoU값을 구해 0.5이상인 경우 positive sample(객체)로, 0.5 미만인 경우 negative sample(배경)으로 저장한다. 그리고 positive sample = 32, negative sample = 96 로 mini batch(=128)을 구성하여 pre-trained된 AlexNet에 입력하여 학습을 진행함. 모델의 output으로 feature vector(2000*4096)이 도출됨.   

### 3. Classification by Linear SVM
##### Linear SVM은 2000*4096 feature vector를 입력받아 class를 예측하고 confidence score를 반환한다. 즉 여기서 Linear SVM은 2000개의 classes와 confidence scores를 반환한다. 이 때 linear SVM 모델은 특정 class에 해당하는지 여부만을 판단하는 이진 분류기(binary classifier)임. 따라서 N개의 class를 예측한다고 할 때, 배경을 포함한 (N+1)개의 독립적인 linear SVM 모델을 학습시켜야 한다.   

##### 먼저 객체와 배경을 모두 학습하기 위해 PASCAL VOC 데이터셋에 Selective search 알고리즘을 적용하여 region proposals를 추출함. AlexNet 모델을 fine tune할 때와는 다르게 오직 ground truth box만을 positive sample로, IoU 값이 0.3 미만인 예측 bounding box를 negative sample로 저장함. IoU 값이 0.3 이상인 bounding box는 무시함. positive sample = 32, negative sample = 96 이 되도록 mini batch(=128)을 구성한 뒤 fine tuned AlexNet에 입력하여 feature vector를 추출하고, 이를 linear SVM에 입력하여 학습시킴. 이 때 하나의 linear SVM 모델은 특정 class에 해당하는지 여부를 학습하기 때문에 output unit = 2 다. 학습이 한 차례 끝난 후, hard negative mining 기법을 적용하여 재학습시킴.

#### Hard Negative Mining
##### Hard negative mining은 모델이 예측에 실패하는 어려운(hard) sample들을 모으는 기법으로, hard negative mining을 통해 수집된 데이터를 활용하여 모델의 결과를 향상시킬 수 있다. 예를들어 이미지에서 사람의 얼굴을 탐지하는 모델을 학습시킬 때, 모델이 안면이라고 예측했지만 실제로 배경인 경우는 False Positive Sample에 해당된다. 이 sample들을 학습 과정에서 추가하여 재학습시키면 모델의 성능이 올라가고 False Positive라고 판단하는 오류가 줄어듬.   

### 4. Detailed localization by Bounding Box Regressor
##### Selective search 알고리즘을 통해 얻은 객체의 위치는 다소 부정확할 수 있다. 이러한 문제를 해결하기 위해 bounding box의 좌표를 변환하여 객체의 위치를 세밀하게 조정해주는 것이 Bounding box regressor모델이다.   

##### PASCAL 데이터셋에 Selective search 알고리즘을 적용하여 얻은 region proposals를 학습 데이터로 사용. 이 때 별도의 negative sample은 정의하지 않고 IoU 값이 0.6 이상인 sample을 positive sample로 정의함. positive sample을 fine tuned된 AlexNet에 입력하여 얻은 feature vector를 Bounding box regressor에 입력하여 학습시킴. Bounding Box Regressor는 feature vector를 입력받아 조정된 bounding box 좌표값(output unit=4)를 반환한다. 결과적으로 2000개의 Region Proposals에 대한 조정된 bounding box 좌표값을 추론함.   

### 5. Non Maximum Supression
##### linear SVM 모델과 Bounding box regressor 모델을 통해 얻은 2000개의 bounding box를 전부 다 표시할 경우 하나의 객체에 대해 지나치게 많은 bounding box가 겹칠 수 있다. 이로 인해 객체 탐지의 정확도가 떨어질 수 있으며 이러한 문제를 해결하기 감지된 bounding box 중에서 비슷한 위치에 있는 box를 제거하고 가장 적합한 box를 선택하는 Non maximum suppression 알고리즘을 적용한다.   

##### 1. bounding box별로 지정한 confidence score threhold 이하의 box를 제거  Ex) confidence score threshold=0.5면 confidence score가 0.5보다 작은 box를 제거함.
##### 2. 남은 bounding box를 confidence score에 따라 내침차순으로 정렬한 후 confidence score가 높은 순의 bounding box부터 다른 box와의 IoU 값을 조사하여 IoU threshold 이상인 box를 모두 제거  Ex) IoU threshold=0.4고 confidence score에 따라 box를 내림차순으로 정렬 한 후, score가 가장 큰 box와 나머지 박스의 IoU값을 구해서 IoU threshold보다 크다면 비교당한 score가 작은 box를 삭제함. 이 과정을 반복
##### 3. 남아있는 box만 선택


### - 배운점
##### R-CNN의 구조를 보면 마지막 FC에 softmax가 적용되어 모델의 output이 classes에 대한 확률로 나오는 줄 알았다. 하지만 outputs = model(inputs)의 결과는 활성화 함수가 적용 되기 전의 결과이다. 즉 outputs은 2000*4096인 feature vector인 것이다. 이 outputs에 loss인 CrossEntropy나 softmax등을 적용해야 R-CNN의 구조에서 보았던 softmax가 적용된 FC가 된다는것을 알았다. 따라서 Linear SVM을 학습시킬 때 fine tune한 AlexNet의 최종 출력노드를 2로 설정했으므로 바로 Hinge Loss를 적용해 모델을 학습시키는 방식으로 진행했다.   

##### 아래 사진은 R-CNN을 이용해 Object Detection을 한 결과
<p align="center"><img src="https://github.com/suhyeong-jeon/R_CNN/assets/70623959/40469b72-7ec7-43e6-86e0-6cdf7a74b104)https://github.com/suhyeong-jeon/R_CNN/assets/70623959/40469b72-7ec7-43e6-86e0-6cdf7a74b104"></p>

### - 참조 링크
##### - <https://herbwood.tistory.com/5>
##### - <https://herbwood.tistory.com/6>
##### - <https://github.com/object-detection-algorithm/R-CNN>

##### 해당 코드는 zjZSTU(<https://github.com/object-detection-algorithm/R-CNN>)를 참조하여 작성했음.
