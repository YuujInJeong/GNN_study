
- 목차

# 📘 GIN (Graph Isomorphism Network) 완전 정리

## 1. 도입: 그래프 표현력의 한계와 WL 테스트

### 🧩 문제의식

기존 GCN, GAT 같은 GNN은 **서로 다른 그래프 구조를 구별하지 못하는 경우**가 있었음

→ GNN의 구조 구별력을 어떻게 평가할 수 있을까?

---

### ✅ Weisfeiler-Lehman (WL) Test

**그래프 동형성 판단을 위한 고전적인 알고리즘**

- **노드의 이웃 레이블을 반복적으로 모아서 색칠**함
- 색이 다르면 → 구조가 다르다고 판단
- 이를 GNN과 연결하면
    
    → GNN이 WL 테스트 수준의 구별력을 가질 수 있냐가 핵심 질문!
    

---

### 🧠 GIN의 출발점

Xu et al. (ICLR 2019)은

> “기존 GNN은 1-WL 테스트보다 약한 표현력을 가진다”
> 
> 
> → 우리는 **1-WL 수준의 표현력을 갖는 GNN**, 즉 GIN을 만들자!
> 

---

## 2. GIN의 핵심 설계: Sum Aggregator + MLP

### 🔄 업데이트 식

hv(k)=MLP(k)((1+ϵ)⋅hv(k−1)+∑u∈N(v)hu(k−1))h_v^{(k)} = \text{MLP}^{(k)}\left((1 + \epsilon) \cdot h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)

| 구성 요소 | 설명 |
| --- | --- |
| ∑\sum | 이웃 정보 집계. 평균 아님! → 구조 정보 손실 ↓ |
| ϵ\epsilon | 자기 정보의 중요도 조절 (learnable scalar) |
| MLP | 복잡한 구조 표현 및 단사 함수 근사 |

---

### 🔍 왜 Mean이 아니라 Sum인가?

| Aggregator | 단사성 | 정보 손실 | 표현력 |
| --- | --- | --- | --- |
| Mean | ❌ | 큼 | 약함 (GCN) |
| Sum | ✅ | 적음 | 강함 (GIN) |
- Sum은 multiset 정보를 보존할 수 있는 유일한 연산
- MLP가 그것을 단사적으로 mapping → 표현력 ↑

---

### 💡 MLP = Universal Approximator

MLP는 이론상 어떤 연속 함수도 근사 가능

→ 희석된 정보를 **recover**할 수 있음

---

## 3. GIN의 표현력 = 1-WL 테스트와 동등

- 논문에서는 수학적으로

> GIN은 1-WL 테스트만큼 구조를 잘 구별할 수 있음
> 
- 기존 GCN은 WL보다 약함 (평균화에 의한 정보 손실 때문)

---

## 4. GIN은 왜 그래프 분류(Graph Classification)에 강할까?

### 🧪 노드 분류 vs 그래프 분류

| 구분 | 노드 분류 | 그래프 분류 |
| --- | --- | --- |
| 예시 | "이 노드는 AI 분야일까?" | "이 분자 그래프는 독성이 있을까?" |
| 중점 | **노드 주변 이웃 (local)** | **전체 구조 (global)** |
| 구조적 민감도 | 중간 | **매우 중요** |

---

### 🎯 GIN의 강점

- GIN은 **노드 이웃의 multiset 구조를 구별** 가능
- GIN은 **sum + MLP**로 다중 hop 구조까지 보존 가능
- 그래서 그래프 전체 구조를 표현하는 데 유리함

hG=READOUT({hv(k)∣v∈G})(e.g., sum pooling)h_G = \text{READOUT}\left(\{ h_v^{(k)} \mid v \in G \} \right)
\quad \text{(e.g., sum pooling)}

→ 즉, GIN은 **노드 단위 표현을 잘 만들어서**,

→ 그것들을 global하게 모아서 **그래프 임베딩**으로 활용 가능

---

## 5. GIN은 **topology 의존적인 네트워크**

> 그래프의 형태와 연결 구조에 민감하게 반응
> 
> 
> → 이게 GIN의 핵심 강점이자 차별점
> 

### 📐 왜 topology-aware한가?

- Sum aggregator는 단순 평균이 아님 → 구조별로 다른 임베딩 생성
- MLP가 이 구조 차이를 recover 가능
- 즉, 노드가 연결된 방식(topology)에 따라 완전히 다른 결과를 냄

> GCN은 [a,a,a]와 [a,b,c] 구분 못하지만,
> 
> 
> GIN은 이걸 구분 가능 (sum 다르고, MLP 다르게 반응)
> 

---

## 6. GIN의 **feature scaling 문제**와 대응

### ⚠️ 문제점: Sum은 degree가 커질수록 값도 커짐

sum([h1,h2])=h1+h2vssum([h1,h2,...,h20])=크기 증가!\text{sum}([h_1, h_2]) = h_1 + h_2  
\quad \text{vs} \quad  
\text{sum}([h_1, h_2, ..., h_{20}]) = \text{크기 증가!}

→ 이웃 수가 많은 노드는

→ feature가 더 커져서 **스케일 불균형 발생**

→ 학습 불안정 가능

---

### 🛠️ 해결 방법

| 방법 | 설명 |
| --- | --- |
| **BatchNorm** | 레이어마다 출력 정규화 (논문에서 권장) |
| **Dropout** | 과도한 노드 영향 방지 |
| **ε 조절** | 자기 정보의 비중을 학습해서 조정 |
| (선택) degree-normalized sum | GCN처럼 normalization 적용 → 표현력 일부 희생 |

---

## 7. 실험 결과 (논문 기반)

Xu et al. (2019)에서는 여러 벤치마크에서 실험했어.

| Dataset | GCN | GAT | **GIN** |
| --- | --- | --- | --- |
| MUTAG | 85.6 | 86.5 | **89.4** |
| PROTEINS | 75.3 | 75.7 | **76.2** |
| NCI1 | 80.2 | 81.1 | **83.9** |
| COLLAB | 79.0 | 79.4 | **80.2** |

→ 특히 **그래프 분류**에서 GIN의 성능이 뚜렷하게 좋았음

---

## ✅ 전체 요약

| 항목 | GIN 특성 |
| --- | --- |
| Aggregator | Sum (단사성 확보, 구조 보존) |
| 표현력 | WL 테스트와 동등 |
| MLP 역할 | Universal approximator로 구조 복원 |
| 구조 민감도 | 매우 높음 (topology-aware) |
| 분류 성능 | 그래프 분류에서 특히 강력함 |
| 단점 | feature scaling 이슈 있음 → BatchNorm 등으로 보완 |