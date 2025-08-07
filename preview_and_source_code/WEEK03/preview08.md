
# 📘 GraphSAGE 학습 정리

## 📌 1. GraphSAGE란 무엇인가?

GraphSAGE는 “Graph Sample and Aggregate”의 줄임말로,

기존 GCN이 갖고 있던 제한적인 학습 구조(transductive learning)를 극복하고,

**새로운 노드(unseen node)에도 일반화가 가능한 inductive learning 방식의 GNN**으로 제안되었다.

2017년 Hamilton et al.이 NeurIPS에 발표한 논문 [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

을 통해 처음 소개되었다.

기존 GCN은 전체 그래프 구조를 입력으로 사용하기 때문에, 학습 시점 이후에 새롭게 등장한 노드나 부분 그래프에 대해 일반화가 어려운 transductive 방식이지만, GraphSAGE는 **샘플링 기반의 local aggregation 방식**을 사용하여 **학습에 등장하지 않은 노드**에 대해서도 유의미한 임베딩을 생성할 수 있다.

이러한 특성 덕분에 Pinterest, Facebook, Amazon 등 실제 대규모 그래프 기반 추천 시스템에 적용된 사례가 존재하며, 웹스케일 GNN의 출발점이 된 모델이다.

---

## 🧠 2. GraphSAGE의 핵심 아이디어

GraphSAGE는 각 노드가 이웃 노드들로부터 정보를 모으는 과정에서 전체 이웃을 사용하지 않고,

**고정된 수의 이웃만 샘플링(sample)**하고, 이들을 특정 집계 함수(aggregator)를 통해 요약한 뒤,

자신의 이전 임베딩과 이웃 요약 결과를 **concat**하여 새로운 임베딩을 생성하는 방식이다.

즉, GraphSAGE는 다음과 같은 연산 흐름을 따른다.

1. **이웃 샘플링**
    - 각 노드는 이웃 중 정해진 수 (예: 10개)만 샘플링한다.
2. **Aggregation**
    - 샘플링된 이웃 노드의 임베딩을 집계함 (Mean, LSTM, Pooling 등 가능)
3. **Combination (Concat)**
    - 집계된 이웃 임베딩과 자기 자신의 임베딩을 이어붙인다.
4. **Linear + ReLU**
    - Weight matrix를 곱한 후 ReLU를 적용한다.
5. **(Optional) ℓ2 정규화**

이 과정을 여러 레이어(k-hop)로 쌓으면 멀리 있는 이웃의 정보까지 반영할 수 있다.

---

## 📐 3. 수식 기반 구조

![스크린샷 2025-07-24 오후 6.25.55.png](attachment:4c46e123-5b79-4739-ab55-60d2b90d2984:스크린샷_2025-07-24_오후_6.25.55.png)

---

## 🧪 4. Aggregator 함수 종류

GraphSAGE는 다양한 aggregator 함수를 실험적으로 도입하였다:

| 종류 | 방식 | 특징 |
| --- | --- | --- |
| **Mean** | 단순 평균 | 계산 간단, 빠름 |
| **LSTM** | 이웃들을 시퀀스로 넣고 LSTM 통과 | 표현력 있음, 순서 임의 |
| **Pooling** | 이웃 각각에 MLP 적용 후 Max Pooling | 중요 정보 강조 |
| **GCN-style** | GCN처럼 정규화된 평균 | GCN과 유사한 구조, 정보 희석 가능성 있음 |

각 aggregator는 그래프 구조나 노드의 피처 특성에 따라 성능 차이를 보일 수 있으며, 실제 구현에서는 가장 간단한 mean aggregator가 널리 사용된다.

---

## 🧱 5. GCNConv와 SAGEConv의 행렬 구조 차이

GraphSAGE와 기존 GCN은 모두 그래프의 이웃 정보를 활용하지만,

내부 연산 방식—특히 **행렬 구성 방식(matrix formulation)**—에서는 큰 차이를 보인다.

![스크린샷 2025-07-24 오후 6.27.07.png](attachment:f4669a6f-a839-4c15-9cbd-4d2371143328:스크린샷_2025-07-24_오후_6.27.07.png)

이 구조는 그래프 전체를 고려하므로, **노드 수 n이 커지면 메모리와 계산량이 크게 증가**한다.

또한, **transductive** 특성상 새로운 노드에 대해서는 학습된 모델을 직접 적용할 수 없다.

---

### SAGEConv (GraphSAGE Layer)의 연산 방식

GraphSAGE는 로컬 이웃 집계 기반의 공간적 연산(spatial)을 수행한다.

즉, 각 노드 단위로 **이웃 노드 일부를 샘플링하고**, 그 집계값과 자기 자신의 임베딩을 concat하여 처리한다.

![스크린샷 2025-07-24 오후 6.27.31.png](attachment:25c44719-b8b2-4509-9334-8b1f2f6e1421:스크린샷_2025-07-24_오후_6.27.31.png)

---

## 🧮 6. 비지도 학습에서의 Loss Function

GraphSAGE는 라벨이 없는 환경에서도 노드 임베딩을 학습할 수 있다.

이때 사용하는 loss function은 **노드 간 유사도 기반의 negative sampling loss**로,

Word2Vec의 Skip-gram 모델과 유사하다.

![스크린샷 2025-07-24 오후 6.28.59.png](attachment:7cab0616-93b9-44e0-be95-390952aaf216:스크린샷_2025-07-24_오후_6.28.59.png)

이 수식은 **이웃 노드끼리는 임베딩을 가깝게**, **무관한 노드끼리는 임베딩을 멀게** 만들도록 유도한다.

즉, 학습 과정에서 그래프 구조를 반영한 유사도 임베딩이 형성된다.

---

## 🔄 7. inductive vs transductive의 실제 차이

GraphSAGE의 가장 큰 장점 중 하나는 **inductive learning**이 가능하다는 것이다.

### transductive (GCN의 경우)

---

- 전체 그래프를 알고 있어야 함
- 학습 시 등장하지 않은 노드 → 임베딩 불가

### inductive (GraphSAGE의 경우)

---

- 새로운 노드가 나타나도, **이웃 피처만 있으면 임베딩 가능**
- 예: Pinterest의 신규 게시물, Facebook의 새 사용자

이는 GraphSAGE가 **모델 파라미터를 이웃 패턴을 통해 학습**하고, 이 패턴을 활용하여 unseen 노드도 처리할 수 있기 때문에 가능하다.

## 📊 8. 실험 성능과 실제 적용 사례

논문에서 제시한 실험 결과에 따르면, GraphSAGE는 **비지도 학습만으로도** 매우 높은 분류 성능을 보여준다.

예를 들어 PPI 데이터셋에서 비지도 방식으로 학습한 후 logistic classifier를 붙였을 때 **F1-score 0.93** 수준의 성능을 보였다. 또한 실제 산업계에서도 다음과 같은 응용이 이루어졌다.

| 사례 | 활용 방식 |
| --- | --- |
| Pinterest | PinSAGE로 발전 → 추천 임베딩 |
| Amazon | 유사 상품 탐색 |
| Facebook | 사용자 행동 모델링, 친구 추천 |

---

## ✅ 마무리 요약

> GraphSAGE는 샘플링과 aggregation 기반의 구조를 통해, 라벨이 없는 그래프에서도 노드 임베딩을 효율적으로 학습하며, 새롭게 등장한 노드에도 일반화 가능한 GNN 모델이다.
> 
> 
> **스펙트럴 방식의 GCN과는 다르게, 공간 기반의 유연한 연산과 미니배치 학습이 가능하여 웹스케일 그래프 처리에도 적합하다.**
>