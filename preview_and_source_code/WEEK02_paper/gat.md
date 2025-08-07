# 📘 Graph Attention Networks (GAT) 

## 1. 연구 문제 정의와 동기 (Problem & Motivation)

### 핵심 문제
> **그래프 구조 데이터에서 노드 간 관계의 중요도를 자동으로 학습하는 문제**
- 기존 GCN의 한계: 모든 이웃 노드에 동일한 가중치 적용
- 고유분해(eigendecomposition) 같은 고비용 연산 필요성

### 기존 연구의 한계점
1. **GCN**: $\tilde{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i||\mathcal{N}_j|}} W^{(l)} \tilde{h}_j^{(l)}\right)$
   - 모든 이웃에 동일한 정규화 가중치 적용
   - 그래프 구조에 의존적 (transductive)

2. **GraphSAGE**: 다양한 집계 함수 사용하지만 이웃별 중요도 구분 없음

### GAT의 해결 방향
- **Self-attention 메커니즘**으로 이웃별 중요도 자동 학습
- **Inductive setting**에서 unseen 그래프에도 적용 가능
- **병렬 계산**으로 효율성 향상

---

## 2. 이론적 배경 및 핵심 개념 (Background & Definitions)

### 그래프 정의
- **그래프**: $G = (V, E)$
  - $V$: 노드 집합, $|V| = N$
  - $E$: 엣지 집합
- **노드 feature**: $\mathbf{h} = \{\tilde{h}_1, \tilde{h}_2, ..., \tilde{h}_N\}$, $\tilde{h}_i \in \mathbb{R}^F$
- **이웃 집합**: $\mathcal{N}_i = \{j \in V : (i,j) \in E\}$

### Attention 메커니즘
- **Attention score**: 노드 쌍 $(i,j)$ 간의 중요도 측정
- **Attention weight**: softmax로 정규화된 최종 가중치
- **Multi-head attention**: 여러 독립적인 attention 메커니즘 병합

### 핵심 가정
- **1-hop 이웃만 고려**: 직접 연결된 노드만 attention 계산
- **마스킹**: adjacency matrix로 연결되지 않은 노드 쌍 제외

---

## 3. 모델 구조 및 핵심 수식 흐름 (Model Architecture & Equations)

### 전체 모델 구조
```
입력 노드 feature → 선형 변환 → Attention score 계산 → 마스킹 → Softmax → 가중합 → 출력
```

### 단일 Attention Head 수식 흐름

#### 3.1 선형 변환 (Linear Transformation)
$$\mathbf{h}'_i = W\mathbf{h}_i, \quad W \in \mathbb{R}^{F' \times F}$$

**역할**: 노드 feature를 더 높은 추상 표현으로 투영

#### 3.2 Attention Score 계산
$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [W\mathbf{h}_i \| W\mathbf{h}_j]\right)$$

**구성 요소**:
- $\mathbf{a} \in \mathbb{R}^{2F'}$: 학습 가능한 attention vector
- $\|$: feature concatenation
- $\text{LeakyReLU}(x) = \max(0.2x, x)$: 음수 기울기 정보 보존

#### 3.3 마스킹 (Masking)
$$e_{ij} = -\infty \quad \text{if } j \notin \mathcal{N}_i$$

**목적**: 연결되지 않은 노드 쌍의 attention score를 0으로 만듦

#### 3.4 Softmax 정규화
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

**특성**: $\sum_{j \in \mathcal{N}_i} \alpha_{ij} = 1$ (확률 분포)

#### 3.5 최종 출력 계산
$$\mathbf{h}'_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W\mathbf{h}_j\right)$$

**해석**: 중요한 이웃일수록 더 큰 가중치로 feature 집계

### Multi-head Attention 확장

#### 3.6 K개 Head 병합 (Concatenation)
$$\mathbf{h}'_i = \|_{k=1}^K \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k\mathbf{h}_j\right)$$

**출력 차원**: $KF'$

#### 3.7 K개 Head 평균 (Averaging)
$$\mathbf{h}'_i = \sigma\left(\frac{1}{K}\sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k\mathbf{h}_j\right)$$

**출력 차원**: $F'$

---

## 4. 핵심 수식의 예시와 해석 (Worked Example)

### 예시 그래프 설정
```
노드: A, B, C (3개)
엣지: A-B, A-C
Feature 차원: F=2, F'=3
```

### 입력 데이터
$$\mathbf{h}_A = [1, 2], \quad \mathbf{h}_B = [3, 1], \quad \mathbf{h}_C = [0, 4]$$

### 가중치 행렬 (예시)
$$W = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad \mathbf{a} = [0.5, 0.3, -0.2]$$

### 단계별 계산

#### 4.1 선형 변환
$$W\mathbf{h}_A = [1, 2, 3], \quad W\mathbf{h}_B = [3, 1, 4], \quad W\mathbf{h}_C = [0, 4, 4]$$

#### 4.2 Attention Score 계산 (노드 A 기준)
$$e_{AB} = \text{LeakyReLU}([0.5, 0.3, -0.2]^T [1,2,3,3,1,4]) = \text{LeakyReLU}(0.5 \cdot 1 + 0.3 \cdot 2 + (-0.2) \cdot 3 + 0.5 \cdot 3 + 0.3 \cdot 1 + (-0.2) \cdot 4) = \text{LeakyReLU}(1.4) = 1.4$$

$$e_{AC} = \text{LeakyReLU}([0.5, 0.3, -0.2]^T [1,2,3,0,4,4]) = \text{LeakyReLU}(0.5 \cdot 1 + 0.3 \cdot 2 + (-0.2) \cdot 3 + 0.5 \cdot 0 + 0.3 \cdot 4 + (-0.2) \cdot 4) = \text{LeakyReLU}(0.9) = 0.9$$

#### 4.3 Softmax 정규화
$$\alpha_{AB} = \frac{\exp(1.4)}{\exp(1.4) + \exp(0.9)} = \frac{4.06}{4.06 + 2.46} = 0.62$$

$$\alpha_{AC} = \frac{\exp(0.9)}{\exp(1.4) + \exp(0.9)} = \frac{2.46}{4.06 + 2.46} = 0.38$$

#### 4.4 최종 출력
$$\mathbf{h}'_A = \sigma(0.62 \cdot [3,1,4] + 0.38 \cdot [0,4,4]) = \sigma([1.86, 2.54, 3.84])$$

---

## 5. 논리적 증명 또는 모델 특성 분석 (Proofs or Analysis)

### 5.1 Permutation Invariance 증명

**명제**: GAT는 노드 순서에 무관하다 (permutation invariant)

**증명**:
1. Attention score: $e_{ij} = f(W\mathbf{h}_i, W\mathbf{h}_j)$
2. 노드 순서 변경 시 $f(W\mathbf{h}_i, W\mathbf{h}_j) = f(W\mathbf{h}_j, W\mathbf{h}_i)$ (대칭성)
3. Softmax는 순서에 무관하므로 $\alpha_{ij}$도 순서에 무관
4. 최종 출력도 순서에 무관

### 5.2 계산 복잡도 분석

**시간 복잡도**: $O(|V|FF' + |E|F')$
- 선형 변환: $O(|V|FF')$
- Attention 계산: $O(|E|F')$ (sparse graph에서 $|E| \ll |V|^2$)

**공간 복잡도**: $O(|V|F' + |E|)$
- 노드 feature: $O(|V|F')$
- Attention weight: $O(|E|)$

### 5.3 Inductive 특성 증명

**명제**: GAT는 unseen 노드에도 적용 가능

**증명**:
1. Attention score는 노드 feature만으로 계산
2. 그래프 구조 정보는 마스킹에만 사용
3. 새로운 노드가 추가되어도 기존 가중치 $W, \mathbf{a}$ 재사용 가능

---

## 6. 실험 설정과 결과 요약 (Experiments & Results)

### 6.1 데이터셋 특성

| 데이터셋 | 노드 수 | 엣지 수 | 클래스 수 | Feature 수 | 학습 방식 |
|---------|---------|---------|-----------|------------|-----------|
| Cora | 2,708 | 5,429 | 7 | 1,433 | Transductive |
| Citeseer | 3,327 | 4,732 | 6 | 3,703 | Transductive |
| Pubmed | 19,717 | 44,338 | 3 | 500 | Transductive |
| PPI | ~2,372 | - | 121 (다중) | 50 | Inductive |

### 6.2 모델 설정

**Transductive 설정**:
- 2-layer GAT
- Layer 1: 8-head attention, ELU activation
- Layer 2: 1-head attention, softmax
- Dropout: 0.6, L2 regularization: 0.0005

**Inductive 설정**:
- 3-layer GAT
- Layer 1-2: 4-head attention
- Layer 3: 6-head attention, logistic sigmoid
- Skip connection, Adam optimizer

### 6.3 성능 결과

| 데이터셋 | GCN | GAT | 향상도 |
|---------|-----|-----|--------|
| Cora | 81.4% | 83.0% | +1.6% |
| Citeseer | 70.3% | 72.5% | +2.2% |
| Pubmed | 79.0% | 79.0% | 0.0% |
| PPI (micro-F1) | 50.0% | 97.3% | +47.3% |

### 6.4 Attention Weight 분석
- t-SNE 시각화에서 클러스터 형성 확인
- 중요한 이웃에 높은 attention weight 할당
- Multi-head attention이 다양한 관계 패턴 포착

---

## 7. 한계점과 향후 방향 (Limitations & Future Directions)

### 7.1 현재 한계점

**계산적 한계**:
- Attention 계산이 $O(|E|F')$로 엣지 수에 비례
- 대규모 그래프에서 메모리 부족 가능성

**구조적 한계**:
- 1-hop 이웃만 고려 (고차 이웃 정보 손실)
- Edge feature 미사용
- Attention weight 해석 어려움

**실용적 한계**:
- Hyperparameter tuning 복잡성
- Attention weight의 불안정성

### 7.2 향후 연구 방향

**확장성 개선**:
- Sparse attention 구현
- Hierarchical attention 구조
- Graph partitioning 기반 병렬화

**기능 확장**:
- Edge feature 통합: $e_{ij} = f(W\mathbf{h}_i, W\mathbf{h}_j, W_e\mathbf{e}_{ij})$
- 고차 이웃 attention: $e_{ij}^{(l)} = f(\mathbf{h}_i^{(l-1)}, \mathbf{h}_j^{(l-1)})$
- Graph-level attention

**해석 가능성**:
- Attention weight 시각화 도구
- 중요 노드/엣지 식별 알고리즘
- Attention pattern 분석

---

## 8. 한눈에 보는 요약 (TL;DR)

**GAT는 self-attention 메커니즘을 그래프에 적용하여 노드 간 관계의 중요도를 자동으로 학습하는 GNN 모델이다. 핵심 수식 $e_{ij} = \text{LeakyReLU}(\mathbf{a}^T[W\mathbf{h}_i \| W\mathbf{h}_j])$로 attention score를 계산하고, $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$로 정규화하여 최종 출력 $\mathbf{h}'_i = \sigma(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W\mathbf{h}_j)$를 생성한다. GCN 대비 1.5-2.2% 성능 향상을 보이며, inductive setting에서도 우수한 일반화 능력을 보인다. 다만 계산 복잡도와 attention weight 해석성 측면에서 개선 여지가 있다.**