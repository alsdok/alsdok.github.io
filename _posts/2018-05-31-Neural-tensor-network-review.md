---
layout: post
title: Reasoning With Neural Tensor Networks for Knowledge Base Completion
summary: Neural Tensor Network 논문 리뷰
---

# Reasoning With Neural Tensor Networks for Knowledge Base Completion
###### Rechard Socher, Danqi Chen, Christopher D. Manning, Andrew Y.Ng
###### Computer Science Department, Stanford University, Sta뷰뷰nford, CA 94305, USA

### 배경
지식 베이스는 Question&Answering 등의 작업의 기반 작용을 한다는 점에서 중요하다.
Neural Tensor Networks는 이러한 지식 베이스 기반의 추론을 위해 Relation에 대한 두 Entity 간의 관계를 학습할 수 있는 모델이다.
예를 들어, '수마트라 호랑이'와 '벵골 호랑이'는 모두 호랑이 과에 속하는 지식 베이스를 기반으로 관계 강도를 형성할 수 있게 된다.
WordNet, Yago, Google Knowledge Graph와 같은 기존 온톨로지와 지식 베이스들은 다양한 분야에서 유용한 지식 베이스로 활용되었으나, 그 밖의 추론 능력을 갖추지 못한다는 한계를 가진다. 즉, 지식 베이스와 매우 유사한 사실이 나타나더라도 확장될 수 없는 지식이다.

[NTN Overview]
{: refdef: style="text-align: center;"}
![Alt text](https://scontent-icn1-1.xx.fbcdn.net/v/t1.0-9/23722362_399750990460902_8320244431357040643_n.jpg?_nc_cat=0&oh=688292628105cd8b7ff9f4828af96872&oe=5C2941BD 'NTN시각화')
{: refdef}

### 목적 및 의도화화
Neural Tensor Network(이하 NTN)는 이러한 기존 지식 베이스의 한계를 극복할 수 있는 모델로써 제안된다.
따라서 NTN의 목적은 기존 지식 베이스에 존재하는 지식만을 활용하여 지식 베이스가 내포하고 있는 것 이외의 확장된 사실을 예측하는 것이 된다. 
NTN의 접근법은, ==지식 베이스에 속한 수많은 Entity를 모두 Vector화 하며, 각각의 Entity들의 관계를 이어주고 있는 Relation에 대하여 얼만큼의 관계성으로 묶일 수 있는지를 추론하는 것이다.==

논문이 의도한 시사점은 다음과 같다.
-- Neural Tensor Networks 라고 하는 일반화된 Neural Network 모델을 제안하였다는 점
-- 지식 베이스에 존재하는 Vector화된 Entity들을 결합하는 방식을 제안하였다는 점
-- 레이블이 존재하지 않는 지식 베이스 데이터로부터 Relation을 기준으로 Entity 간의 관계 정확도를 높여갈 수 있는 학습 모델을 제안하였다는 점

### 모델
Neural Tensor Network는 지식 베이스 내에 존재하는 수많은 Vector화된 Entity(혹은 Entity의 결합)를 특정 Relation을 기준으로 연결하였을 때 그 관계에 대해 Score를 반환하는 모델로써 구성된다. 따라서 학습 과정 이후에 이상적인 결과는 특정 Relation을 기준으로 관계성이 높은 Entity 들의 조합은 높은 Score를 가지게 된다.

[NTN 모델 그림 및 수식]
{: refdef: style="text-align: center;"}
![Alt text](https://scontent-icn1-1.xx.fbcdn.net/v/t1.0-9/23659303_399751007127567_2173591880411123681_n.jpg?_nc_cat=0&oh=b8f74957c33f5f9f6e9ad05ad2a6c343&oe=5C3AE539 'NTN 수식')
{: refdef}


NTN의 Activation Function은 Tanh를 사용하고 있다. e1과 e2 두 Entity를 텐서 레이어와의 연산으로 연결하고 있으며 Standard Layer는 두 Entity Vector가 Concat된 형태에 연산되고 더해진다. NTN의 결과 값은 관계에 대한 Score이므로, Shape을 제한하는 용도로 가중치 U를 사용하고 있다.

텐서 레이어의 Slice는 Tensor layer를 얼마만큼 펼쳐서 연산할 것인가에 대한 하이퍼파라미터로, Slice가 커질수록 더 많은 가중치를 사용하게 되며 복잡한 모델이 된다.(일반적인 Neural Network에서 사용하는 Hidden Layer 수나 Hidden Layer 상의 Node 수의 역할) 반면 Tensor Layer에서 Slice의 크기가 정해지면, Standard Layer는 두 Entity를 Concat한 형태에 연산이 이루어져야 하므로 [slice size, 2*Entity 벡터의 길이]로 고정된다. 마지막으로 U의 Transpose 형태는 [1, slice]로 지정되어 최종 Score 값의 Shape을 조정하게 된다.

### 학습
Loss function은 Margin-Max loss를 사용하여 정답 데이터와 오답 데이터(Corrupted triplet)의 스코어 간격을 최대화하도록 학습이 진행된다. 여기서 오답 데이터를 생성하는 방법으로 Entity2를 학습 데이터 내의 랜덤 vector로 매칭하는 방식을 사용하게 된다. 따라서 정답셋 하나에 대한 오답셋의 수도 사용자가 지정하는 하이퍼파라미터이다. 마지막으로 L2 정규화항을 추가하여 최종 Loss값을 계산한다.


