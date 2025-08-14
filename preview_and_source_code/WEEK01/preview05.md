### 5장. 기본 인공신경망을 사용한 노드 특성값 포함시키기 (Including Node Features with Vanilla Neural Networks)

#### 📊 1. 그래프 데이터셋 소개 (Introducing graph datasets)

**사용할 두 데이터셋:**
- **Cora 데이터셋**
- **Facebook Page-Page 데이터셋**

##### 📚 Cora 데이터셋
2008년 Sen 등에 의해 소개된 Cora는 과학 문헌에서 노드 분류를 위한 가장 인기 있는 데이터셋입니다. 2,708개의 논문 네트워크를 나타내며, 각 연결은 참조를 의미합니다.

**Cora 데이터셋 특징**
- 노드: 2,708개의 논문
- 에지: 인용 관계
- 노드 특성: 1,433개 고유 단어의 이진 벡터 (0: 단어 없음, 1: 단어 있음)
- 목표: 7개 카테고리 중 하나로 각 노드 분류
- 표현: 자연어 처리에서 이진 bag of words라고도 불림

**시각화**
그래프가 너무 커서 networkx 같은 Python 라이브러리로 시각화하기 어려운 경우가 많습니다. 이를 위해 전용 도구들이 개발되었습니다: **yEd Live** (https://www.yworks.com/yed-live/), **Gephi** (https://gephi.org/)

```
Cora 데이터셋 시각화 (yEd Live)
- 주황색 노드: 논문
- 초록색 연결: 인용 관계
- 클러스터 형성: 상호 연결된 논문들
```

**PyTorch Geometric으로 데이터셋 로드**

```python
from torch_geometric.datasets import Planetoid

# 데이터셋 다운로드
dataset = Planetoid(root=".", name="Cora")
data = dataset[0]

# 데이터셋 정보 출력
print(f'Dataset: {dataset}')
print('---------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
```

출력:
```
Dataset: Cora()
---------------
Number of graphs: 1
Number of nodes: 2708
Number of features: 1433
Number of classes: 7
```

**그래프 속성 확인**

```python
print(f'Graph:')
print('------')
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')
```

출력:
```
Graph:
------
Edges are directed: False
Graph has isolated nodes: False
Graph has loops: False
```

##### 📘 Facebook Page-Page 데이터셋

2019년 Rozemberczki 등에 의해 소개된 이 데이터셋은 2017년 11월 Facebook Graph API를 사용하여 생성되었습니다.

**Facebook Page-Page 데이터셋 특징**
- 노드: 22,470개의 공식 Facebook 페이지
- 에지: 상호 좋아요 관계
- 노드 특성: 페이지 소유자가 작성한 텍스트 설명에서 생성된 128차원 벡터
- 목표: 4개 카테고리로 분류 (정치인, 기업, TV 프로그램, 정부기관)

**Cora와의 주요 차이점**
- 노드 수가 훨씬 많음 (2,708 vs 22,470)
- 노드 특성 차원이 크게 감소 (1,433 → 128)
- 분류 카테고리가 적음 (7개 → 4개, 더 쉬운 작업)

**데이터셋 로드**

```python
from torch_geometric.datasets import FacebookPagePage

# 데이터셋 다운로드
dataset = FacebookPagePage(root=".")
data = dataset[0]

# 데이터셋 정보 출력
print(f'Dataset: {dataset}')
print('-----------------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
```

출력:
```
Dataset: FacebookPagePage()
-----------------------
Number of graphs: 1
Number of nodes: 22470
Number of features: 128
Number of classes: 4
```

**그래프 속성 확인**

```python
print(f'\nGraph:')
print('------')
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')
```

출력:
```
Graph:
------
Edges are directed: False
Graph has isolated nodes: False
Graph has loops: True
```

**마스크 생성**
Facebook Page-Page 데이터셋은 기본적으로 훈련, 검증, 테스트 마스크가 없으므로 임의로 생성해야 합니다:

```python
# 임의로 마스크 생성
data.train_mask = range(18000)
data.val_mask = range(18001, 20000)
data.test_mask = range(20001, 22470)
```

---

#### 🧮 2. 기본 신경망으로 노드 분류 (Classifying nodes with vanilla neural networks)

Zachary's Karate Club와 비교하여 이 두 데이터셋은 새로운 유형의 정보인 **노드 특성**을 포함합니다. 이는 소셜 네트워크에서 사용자의 나이, 성별, 관심사 등과 같은 노드에 대한 추가 정보를 제공합니다.

**표 형태 데이터로 취급**
노드 특성은 표 형태 데이터셋으로 쉽게 접근할 수 있습니다. data.x(노드 특성 포함)와 data.y(각 노드의 클래스 라벨)를 병합하여 일반적인 pandas DataFrame으로 변환할 수 있습니다.

```python
import pandas as pd

# Cora 데이터셋을 DataFrame으로 변환
df_x = pd.DataFrame(data.x.numpy())
df_x['label'] = pd.DataFrame(data.y)
```

```
표 형태 표현:
     0    1    ...  1432  label
0    0    0    ...    0      3
1    0    0    ...    0      4
...  ...  ...  ...  ...    ...
2707 0    0    ...    0      3
```

**정확도 함수 정의**

```python
def accuracy(y_pred, y_true):
    return torch.sum(y_pred == y_true) / len(y_true)
```

##### 🏗️ MLP 클래스 구현

**필요한 라이브러리 임포트**

```python
import torch
from torch.nn import Linear
import torch.nn.functional as F
```

**MLP 클래스 구조**

```python
class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.linear1 = Linear(dim_in, dim_h)
        self.linear2 = Linear(dim_h, dim_out)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)
    
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), 
                                   lr=0.01, weight_decay=5e-4)
        
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x)
            loss = criterion(out[data.train_mask], 
                           data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), 
                         data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], 
                                   data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), 
                                 data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | '
                      f'Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc*100:.2f}%')
    
    def test(self, data):
        self.eval()
        out = self(data.x)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], 
                      data.y[data.test_mask])
        return acc
```

##### 📈 MLP 학습 및 평가

**Cora 데이터셋에서 MLP 훈련**

```python
# MLP 모델 생성
mlp = MLP(dataset.num_features, 16, dataset.num_classes)
print(mlp)

# 모델 훈련
mlp.fit(data, epochs=100)

# 테스트 정확도 평가
acc = mlp.test(data)
print(f'MLP test accuracy: {acc*100:.2f}%')
```

출력:
```
MLP(
  (linear1): Linear(in_features=1433, out_features=16, bias=True)
  (linear2): Linear(in_features=16, out_features=7, bias=True)
)

Epoch   0 | Train Loss: 1.954 | Train Acc: 14.29% | Val Loss: 1.93 | Val Acc: 30.80%
Epoch  20 | Train Loss: 0.120 | Train Acc: 100.00% | Val Loss: 1.42 | Val Acc: 49.40%
Epoch  40 | Train Loss: 0.015 | Train Acc: 100.00% | Val Loss: 1.46 | Val Acc: 50.40%
Epoch  60 | Train Loss: 0.008 | Train Acc: 100.00% | Val Loss: 1.44 | Val Acc: 53.40%
Epoch  80 | Train Loss: 0.008 | Train Acc: 100.00% | Val Loss: 1.40 | Val Acc: 54.60%
Epoch 100 | Train Loss: 0.009 | Train Acc: 100.00% | Val Loss: 1.39 | Val Acc: 54.20%

ACC는 높아야하구~ Loss는 낮아야하니~

MLP test accuracy: 52.50%
```

**Facebook Page-Page 데이터셋에서 MLP 훈련**

```python
# Facebook 데이터셋에서 동일한 과정 반복
mlp_facebook = MLP(dataset.num_features, 16, dataset.num_classes)
mlp_facebook.fit(data, epochs=100)
acc_facebook = mlp_facebook.test(data)
print(f'MLP test accuracy: {acc_facebook*100:.2f}%')
```

출력:
```
Epoch   0 | Train Loss: 1.398 | Train Acc: 23.94% | Val Loss: 1.40 | Val Acc: 24.21%
Epoch  20 | Train Loss: 0.652 | Train Acc: 74.52% | Val Loss: 0.67 | Val Acc: 72.64%
Epoch  40 | Train Loss: 0.577 | Train Acc: 77.07% | Val Loss: 0.61 | Val Acc: 73.84%
Epoch  60 | Train Loss: 0.550 | Train Acc: 78.30% | Val Loss: 0.60 | Val Acc: 75.09%
Epoch  80 | Train Loss: 0.533 | Train Acc: 78.89% | Val Loss: 0.60 | Val Acc: 74.79%
Epoch 100 | Train Loss: 0.520 | Train Acc: 79.49% | Val Loss: 0.61 | Val Acc: 74.94%

MLP test accuracy: 74.52%
```

---

#### 🎯 3. 기본 그래프 신경망으로 노드 분류 (Classifying nodes with vanilla graph neural networks)

잘 알려진 GNN 아키텍처를 직접 소개하는 대신, GNN 뒤에 있는 사고 과정을 이해하기 위해 우리만의 모델을 구축해보겠습니다.

##### 🔄 그래프 선형 층의 개념

**기본 신경망 층**
기본 신경망 층은 선형 변환 h = xW에 해당하며, 여기서 x는 노드의 입력 벡터이고 W는 가중치 행렬입니다.

**그래프 맥락의 중요성**
노드 특성만으로는 그래프를 잘 이해할 수 없습니다. 이미지의 픽셀처럼, 노드의 맥락이 중요합니다. 노드를 이해하려면 이웃을 살펴야 합니다.

**그래프 선형 층 수식**
노드 i의 이웃 집합을 N(i)라고 하면, 그래프 선형 층은 다음과 같이 작성할 수 있습니다:

```
h_i = Σ x_j W^T
     j ∈ N(i)
```

**행렬 형태로 변환**
더 효율적인 행렬 곱셈을 위해 다음과 같이 다시 작성할 수 있습니다:

```
H = ÃX
```

여기서 Ã = A + I (인접 행렬 + 자기 루프)

##### 🏗️ VanillaGNNLayer 구현

```python
class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = Linear(dim_in, dim_out, bias=False)
    
    def forward(self, x, adjacency):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)
        return x
```

**인접 행렬 준비**

```python
from torch_geometric.utils import to_dense_adj

# 에지 인덱스를 밀집 인접 행렬로 변환
adjacency = to_dense_adj(data.edge_index)[0]
# 자기 루프 추가
adjacency += torch.eye(len(adjacency))
```

##### 🏗️ VanillaGNN 구현

```python
class VanillaGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gnn1 = VanillaGNNLayer(dim_in, dim_h)
        self.gnn2 = VanillaGNNLayer(dim_h, dim_out)
    
    def forward(self, x, adjacency):
        h = self.gnn1(x, adjacency)
        h = torch.relu(h)
        h = self.gnn2(h, adjacency)
        return F.log_softmax(h, dim=1)
    
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), 
                                   lr=0.01, weight_decay=5e-4)
        
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x, adjacency)
            loss = criterion(out[data.train_mask], 
                           data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), 
                         data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], 
                                   data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), 
                                 data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | '
                      f'Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc*100:.2f}%')
    
    def test(self, data):
        self.eval()
        out = self(data.x, adjacency)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], 
                      data.y[data.test_mask])
        return acc
```

##### 📈 VanillaGNN 학습 및 평가

**Cora 데이터셋에서 GNN 훈련**

```python
# GNN 모델 생성 및 훈련
gnn = VanillaGNN(dataset.num_features, 16, dataset.num_classes)
print(gnn)
gnn.fit(data, epochs=100)
acc = gnn.test(data)
print(f'\nGNN test accuracy: {acc*100:.2f}%')
```

출력:
```
VanillaGNN(
  (gnn1): VanillaGNNLayer(
    (linear): Linear(in_features=1433, out_features=16, bias=False)
  )
  (gnn2): VanillaGNNLayer(
    (linear): Linear(in_features=16, out_features=7, bias=False)
  )
)

Epoch   0 | Train Loss: 2.008 | Train Acc: 20.00% | Val Loss: 1.96 | Val Acc: 23.40%
Epoch  20 | Train Loss: 0.047 | Train Acc: 100.00% | Val Loss: 2.04 | Val Acc: 74.60%
Epoch  40 | Train Loss: 0.004 | Train Acc: 100.00% | Val Loss: 2.49 | Val Acc: 75.20%
Epoch  60 | Train Loss: 0.002 | Train Acc: 100.00% | Val Loss: 2.61 | Val Acc: 74.60%
Epoch  80 | Train Loss: 0.001 | Train Acc: 100.00% | Val Loss: 2.61 | Val Acc: 75.20%
Epoch 100 | Train Loss: 0.001 | Train Acc: 100.00% | Val Loss: 2.56 | Val Acc: 75.00%

GNN test accuracy: 76.80%
```

#### 🏆 성능 비교 및 결과 분석

**100회 반복 실험 결과**

| 데이터셋 | MLP | GNN |
|----------|-----|-----|
| **Cora** | 53.47% (±1.81%) | 74.98% (±1.50%) |
| **Facebook** | 75.21% (±0.40%) | 84.85% (±1.68%) |

**결과 분석**

MLP는 Cora에서 낮은 정확도를 보이지만, Facebook Page-Page 데이터셋에서는 더 나은 성능을 보입니다. 그러나 두 경우 모두 우리의 기본 GNN에 의해 능가됩니다. 이러한 결과는 노드 특성에 토폴로지 정보를 포함시키는 것의 중요성을 보여줍니다.

**GNN의 우수성**
- 표 형태 데이터 대신 GNN은 각 노드의 전체 이웃을 고려
- 이 예제에서 10-20%의 정확도 향상
- 네트워크 구조 정보를 효과적으로 활용

**GNN의 실용적 적용**
- 신약 개발: 새로운 항생제 발견
- 추천 시스템: 사용자-상품 관계 분석
- 교통 예측: 다양한 경로와 교통 수단 관계 고려


https://www.yworks.com/yed-live/

https://gephi.org/
-> 설치헤야함


# VanillaGNN 구조 분석

## ✅ 1. 모델 개요

```python
class VanillaGNN(torch.nn.Module):
```

이 클래스는 PyTorch의 기본 신경망 모듈인 `torch.nn.Module`을 상속한 **2-layer GNN**입니다. 

### 🎯 핵심 아이디어
**VanillaGNN은 기본적으로 Dense Layer에 Adjacency Layer를 추가한 구조**입니다.

```python
VanillaGNN (
  (gnn1): VanillaGNNLayer(
    (linear): Linear(in_features=1433, out_features=16, bias=False)
  )
  (gnn2): VanillaGNNLayer(
    (linear): Linear(in_features=16, out_features=7, bias=False)
  )
)
```

### 🏗️ 전체 구조 다이어그램

```mermaid
graph TD
    A[입력: X, A] --> B[GNN Layer 1]
    B --> C[ReLU Activation]
    C --> D[GNN Layer 2]
    D --> E[Log Softmax]
    E --> F[출력: 노드별 클래스 확률]
    
    subgraph "입력 데이터"
        A1[X: 노드 특징 행렬<br/>2798 × 1433] 
        A2[A: 인접행렬<br/>2798 × 2798]
    end
    
    subgraph "GNN Layer 1"
        B1[A @ X<br/>2798 × 1433]
        B2[× W1<br/>1433 × 16]
        B3[출력: 2798 × 16]
    end
    
    subgraph "GNN Layer 2"
        D1[A @ H1<br/>2798 × 16]
        D2[× W2<br/>16 × 7]
        D3[출력: 2798 × 7]
    end
```

---

## ✅ 2. 입력 데이터 구조

### 📊 데이터 차원 정보
- **노드 수**: 2,798개
- **입력 특징 차원**: 1,433개
- **출력 클래스 수**: 7개

### 🔢 행렬 차원 정리

| 구성 요소 | 차원 | 설명 |
|---------|------|------|
| **X (노드 특징)** | 2798 × 1433 | 각 노드의 1433개 특징 |
| **A (인접행렬)** | 2798 × 2798 | 그래프 연결 정보 |
| **W1 (가중치1)** | 1433 × 16 | 입력 → 은닉층 |
| **W2 (가중치2)** | 16 × 7 | 은닉층 → 출력층 |

### 🎯 핵심 수식: `A^T × W`
- **A^T**: 인접행렬의 전치 (2798 × 2798)
- **W**: 가중치 행렬 (1433 × 16)
- **연산**: `A^T @ X @ W` 형태로 수행

---

## ✅ 3. 네트워크 아키텍처

```python
self.gnn1 = VanillaGNNLayer(dim_in, dim_h)    # 1433 → 16
self.gnn2 = VanillaGNNLayer(dim_h, dim_out)   # 16 → 7
```

### 🏛️ 레이어 구성 다이어그램

```mermaid
graph LR
    subgraph "입력층"
        X[노드 특징<br/>2798 × 1433]
        A[인접행렬<br/>2798 × 2798]
    end
    
    subgraph "은닉층"
        H1[GNN Layer 1<br/>2798 × 16]
        R[ReLU]
    end
    
    subgraph "출력층"
        H2[GNN Layer 2<br/>2798 × 7]
        S[Log Softmax]
    end
    
    X --> H1
    A --> H1
    H1 --> R
    R --> H2
    A --> H2
    H2 --> S
```

---

## ✅ 4. Forward 연산 과정

```python
def forward(self, x, adjacency):
    h = self.gnn1(x, adjacency)     # 1단계: 인접행렬과 특징을 곱함
    h = torch.relu(h)               # 비선형 활성화
    h = self.gnn2(h, adjacency)     # 2단계: 다시 메시지 전달
    return F.log_softmax(h, dim=1)  # 최종 출력: 노드 분류를 위한 log_softmax
```

### 🔄 연산 흐름 상세

```mermaid
graph TD
    A[X: 2798×1433] --> D[GNN Layer 1]
    B[A: 2798×2798] --> D
    C[W1: 1433×16] --> D
    D --> E[H1: 2798×16]
    E --> F[ReLU]
    F --> G[H1_activated: 2798×16]
    G --> H[GNN Layer 2]
    B --> H
    I[W2: 16×7] --> H
    H --> J[H2: 2798×7]
    J --> K[Log Softmax]
    K --> L[출력: 2798×7]
```

### 📐 수식 표현

**Layer 1**: `H₁ = ReLU(A × X × W₁ + b₁)`
- `A × X`: 2798×2798 × 2798×1433 = 2798×1433
- `× W₁`: 2798×1433 × 1433×16 = 2798×16

**Layer 2**: `H₂ = A × H₁ × W₂ + b₂`
- `A × H₁`: 2798×2798 × 2798×16 = 2798×16
- `× W₂`: 2798×16 × 16×7 = 2798×7

**최종 출력**: `Z = log_softmax(H₂)`

---

## ✅ 5. VanillaGNNLayer 상세 분석

### 🧩 Layer 구조

```python
class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = torch.nn.Linear(dim_in, dim_out, bias=False)  # bias=False 주목!

    def forward(self, x, adjacency):
        out = torch.matmul(adjacency, x)        # A @ X
        out = self.linear(out)                  # (A @ X) @ W
        return out
```

### 🔍 메시지 전달 과정

```mermaid
graph LR
    subgraph "Aggregation"
        A[인접행렬 A<br/>2798×2798] 
        X[노드 특징 X<br/>2798×1433]
        A --> M[A @ X<br/>이웃 정보 집계]
        X --> M
    end
    
    subgraph "Transformation"
        M --> T[× W<br/>선형 변환]
        T --> O[출력<br/>2798×16]
    end
```

### 💡 핵심 아이디어

> **이웃 노드의 정보를 모아서(aggregate) W로 투사한다.**

- **Aggregation**: `A @ X` - 각 노드가 이웃 노드들의 특징을 평균/합산
- **Transformation**: `@ W` - 집계된 정보를 새로운 특징 공간으로 투사

---

## ✅ 6. 학습 과정

### 🎯 손실 함수와 옵티마이저

```python
def fit(self, data, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
```

### 🔄 학습 루프

```python
for epoch in range(epochs+1):
    self.train()
    optimizer.zero_grad()
    out = self(data.x, adjacency)  # forward pass
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()                # backward pass
    optimizer.step()               # 가중치 업데이트
```

### 📊 학습 과정 다이어그램

```mermaid
graph TD
    A[입력 데이터] --> B[Forward Pass]
    B --> C[예측값 계산]
    C --> D[손실 계산]
    D --> E[Backward Pass]
    E --> F[그래디언트 계산]
    F --> G[가중치 업데이트]
    G --> H{에포크 완료?}
    H -->|No| B
    H -->|Yes| I[테스트]
```

### 🧪 테스트 함수

```python
def test(self, data):
    self.eval()
    out = self(data.x, adjacency)
    acc = accuracy(out[data.test_mask], data.y[data.test_mask])
    return acc
```

---

## ✅ 7. 실험 결과 및 분석

### 📈 학습 결과

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 0     | 1.991      | 15.71%    | 2.11     | 9.40%   |
| 20    | 0.065      | 99.29%    | 1.47     | 76.80%  |
| 40    | 0.014      | 100.00%   | 2.11     | 75.40%  |
| 60    | 0.007      | 100.00%   | 2.22     | 75.40%  |
| 80    | 0.004      | 100.00%   | 2.20     | 76.80%  |
| 100   | 0.003      | 100.00%   | 2.19     | 77.00%  |

**최종 테스트 정확도: 76.60%**

### 🚨 과적합(Overfitting) 분석

```mermaid
graph LR
    subgraph "과적합 패턴"
        A[Train Loss: 지속적 감소<br/>0.003까지] 
        B[Train Acc: 100% 달성]
        C[Val Loss: 증가 추세<br/>2.19까지]
        D[Val Acc: 정체<br/>77% 수준]
    end
```

#### 🔍 과적합 증상
1. **Train Loss**: 1.991 → 0.003 (지속적 감소)
2. **Train Accuracy**: 15.71% → 100% (완벽한 학습)
3. **Validation Loss**: 2.11 → 2.19 (증가 추세)
4. **Validation Accuracy**: 9.40% → 77% (정체)

#### 💡 과적합 원인
- **모델 복잡도**: 2-layer GNN이 데이터에 비해 복잡
- **데이터 부족**: 학습 데이터가 충분하지 않음
- **정규화 부족**: Dropout이나 더 강한 weight decay 필요

#### 🛠️ 개선 방안
1. **Early Stopping**: Validation Loss 증가 시점에서 학습 중단
2. **Dropout 추가**: 과적합 방지
3. **Weight Decay 증가**: 5e-4 → 1e-3
4. **모델 단순화**: 은닉층 차원 축소

---

## ✅ 8. MLP vs GNN 비교

| 항목      | MLP                  | GNN                        |
| ------- | -------------------- | -------------------------- |
| 연결성     | 노드 간 정보 공유 없음        | 이웃 노드와 정보 공유 (메시지 전달)      |
| 입력 구조   | Dense feature matrix | Feature + adjacency matrix |
| 연산      | `x @ W`              | `A @ x @ W`                |
| 파라미터 공유 | 없음                   | 있음 (W는 여러 노드에 공유됨)         |

### 🔍 핵심 차이점 다이어그램

```mermaid
graph TB
    subgraph "MLP"
        M1[노드 1] --> M2[개별 처리]
        M3[노드 2] --> M4[개별 처리]
        M5[노드 3] --> M6[개별 처리]
    end
    
    subgraph "GNN"
        G1[노드 1] --> G2[이웃 정보 집계]
        G3[노드 2] --> G2
        G4[노드 3] --> G2
        G2 --> G5[공유 가중치로 변환]
    end
```

---

## ✅ 9. 옵티마이저 정리

### 🎯 주요 옵티마이저 비교

| 옵티마이저     | 특징                             |
| --------- | ------------------------------ |
| GD        | 모든 데이터를 사용, 계산량 많음             |
| SGD       | 샘플 단위로 빠르게 업데이트                |
| Momentum  | 진동 감소, 지역 최소 탈출 도움             |
| NAG       | 미래 예측으로 더 정교한 업데이트             |
| Adagrad   | 파라미터별 적응적 학습률, 그러나 감소 과다       |
| RMSprop   | 최근 업데이트 중심, 안정적 학습 유지          |
| **Adam**  | **Momentum + RMSprop + 보정, 널리 사용** |
| Adabelief | Adam 변형 (간단 언급만)               |

### 🏆 VanillaGNN에서 사용하는 Adam

```python
optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
```

- **학습률**: 0.01
- **Weight Decay**: 5e-4 (L2 정규화)
- **장점**: 자동으로 학습률 조정, 빠른 수렴, 과적합 방지

---

## 📚 참고 자료

- [Optimizer 종류 및 정리](https://velog.io/@chang0517/Optimizer-%EC%A2%85%EB%A5%98-%EB%B0%8F-%EC%A0%95%EB%A6%AC)

