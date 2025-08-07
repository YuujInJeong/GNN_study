# 📘 Semi-Supervised Classification with Graph Convolutional Networks (GCN)
---

## 1. 연구 문제 정의와 동기 (Problem & Motivation)

### 핵심 문제
- **그래프 구조 데이터에서 semi-supervised 노드 분류를 효율적으로 수행하는 문제**
- 기존 spectral graph convolution의 고비용 연산 문제 해결
- 그래프 구조와 노드 feature를 동시에 활용하는 end-to-end 학습 방법 제안

### 기존 연구의 한계점
1. **Spectral Graph Convolution**: 
   - 고유분해(eigendecomposition) 기반으로 $O(N^3)$ 계산 복잡도
   - 그래프 구조가 고정되어 있어 inductive setting에서 적용 불가
   - 전체 그래프의 Laplacian matrix 필요

2. **Spatial Methods**:
   - 이웃 노드의 가중치가 고정되어 있음
   - 노드 차수에 따른 정규화 문제
   - 표현력 제한

3. **Semi-supervised Learning**:
   - 그래프 구조 정보 활용 부족
   - 노드 간 관계 정보 손실

### GCN의 해결 방향
- **Chebyshev 다항식 근사**로 계산 복잡도 단순화
- **Renormalization trick**으로 수치적 안정성 확보
- **Layer-wise propagation rule**로 end-to-end 학습 가능
- **Semi-supervised setting**에서 라벨된 노드 정보 전파

---

## 2. 이론적 배경 및 핵심 개념 (Background & Definitions)

### 그래프 정의
- **그래프**: $G = (V, E)$
  - $V$: 노드 집합, $|V| = N$
  - $E$: 엣지 집합
- **Adjacency Matrix**: $A \in \mathbb{R}^{N \times N}$
- **Degree Matrix**: $D_{ii} = \sum_j A_{ij}$
- **Laplacian Matrix**: $L = D - A$
- **Normalized Laplacian**: $\tilde{L} = I_N - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$

### Spectral Graph Theory
- **Graph Fourier Transform**: $f \rightarrow \hat{f} = U^T f$
- **Inverse Graph Fourier Transform**: $\hat{f} \rightarrow f = U\hat{f}$
- **Spectral Convolution**: $g_\theta * x = Ug_\theta U^Tx$

### Chebyshev Polynomial
- **Chebyshev 다항식**: $T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$
- **근사**: $g_\theta(\Lambda) \approx \sum_{k=0}^K \theta_k T_k(\tilde{\Lambda})$
- **K=1 근사**: $g_\theta * x \approx \theta_0 x + \theta_1 \tilde{L}x$

### 핵심 가정
- **1-hop 이웃만 고려**: $K=1$ Chebyshev 근사
- **Self-loop 추가**: $\tilde{A} = A + I_N$
- **Symmetric normalization**: $\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$

---

## 3. 모델 구조 및 핵심 수식 흐름 (Model Architecture & Equations)

### 전체 모델 구조
```
입력 노드 feature → Graph Convolution Layer → 비선형 활성화 → Graph Convolution Layer → Softmax → 출력
```

### Graph Convolution Layer 수식 흐름

#### 3.1 기본 Spectral Convolution
$$g_\theta * x = Ug_\theta(\Lambda)U^Tx$$

**문제점**: 고유분해 $O(N^3)$ 계산 복잡도

#### 3.2 Chebyshev 다항식 근사
$$g_\theta * x \approx \sum_{k=0}^K \theta_k T_k(\tilde{L})x$$

**특성**: 
- $\tilde{L} = \frac{2}{\lambda_{max}}L - I_N$ (정규화)
- $T_k(\tilde{L}) = 2\tilde{L}T_{k-1}(\tilde{L}) - T_{k-2}(\tilde{L})$
- $K=1$ 근사: $g_\theta * x \approx \theta_0 x + \theta_1 \tilde{L}x$

#### 3.3 K=1 근사 및 단순화
$$g_\theta * x \approx \theta_0 x + \theta_1 \tilde{L}x = \theta_0 x + \theta_1(I_N - D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x$$

**단순화**: $\theta_0 = -\theta_1 = \theta$로 설정
$$g_\theta * x \approx \theta(I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x$$

#### 3.4 Renormalization Trick
**문제**: $I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$의 수치적 불안정성

**해결**: $\tilde{A} = A + I_N$, $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$

**최종 수식**:
$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$

### Layer-wise Propagation Rule

#### 3.5 단일 레이어 수식
$$H^{(l+1)} = \sigma\left(\hat{A}H^{(l)}W^{(l)}\right)$$

**구성 요소**:
- $\hat{A} = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$: 정규화된 인접 행렬
- $H^{(l)} \in \mathbb{R}^{N \times d_l}$: $l$번째 레이어의 노드 feature
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$: 학습 가능한 가중치 행렬
- $\sigma(\cdot)$: 비선형 활성화 함수 (ReLU)

#### 3.6 2-layer GCN 구조
$$Z = f(X, A) = \text{softmax}\left(\hat{A} \text{ReLU}\left(\hat{A}XW^{(0)}\right)W^{(1)}\right)$$

**입력**: $X \in \mathbb{R}^{N \times C}$ (노드 feature)
**출력**: $Z \in \mathbb{R}^{N \times F}$ (노드 클래스 확률)

---

## 4. 핵심 수식의 예시와 해석 (Worked Example)

### 예시 그래프 설정
```
노드: A, B, C (3개)
엣지: A-B, B-C
Feature 차원: C=2, Hidden=3, Output=2
```

### 입력 데이터
$$X = \begin{bmatrix} 1 & 2 \\ 3 & 1 \\ 0 & 4 \end{bmatrix}, \quad A = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

### 단계별 계산

#### 4.1 Self-loop 추가
$$\tilde{A} = A + I_3 = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 1 \end{bmatrix}$$

#### 4.2 Degree matrix 계산
$$\tilde{D} = \begin{bmatrix} 2 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 2 \end{bmatrix}$$

#### 4.3 정규화된 인접 행렬
$$\hat{A} = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}} = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} & 0 \\ \frac{1}{\sqrt{6}} & \frac{1}{3} & \frac{1}{\sqrt{6}} \\ 0 & \frac{1}{\sqrt{6}} & \frac{1}{\sqrt{2}} \end{bmatrix}$$

#### 4.4 가중치 행렬 (예시)
$$W^{(0)} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}, \quad W^{(1)} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

#### 4.5 첫 번째 레이어 계산
$$H^{(1)} = \text{ReLU}(\hat{A}XW^{(0)}) = \text{ReLU}\left(\begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} & 0 \\ \frac{1}{\sqrt{6}} & \frac{1}{3} & \frac{1}{\sqrt{6}} \\ 0 & \frac{1}{\sqrt{6}} & \frac{1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 1 \\ 0 & 4 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}\right)$$

$$= \text{ReLU}\left(\begin{bmatrix} 2.45 & 1.63 & 4.08 \\ 1.63 & 1.09 & 2.72 \\ 1.22 & 2.45 & 3.67 \end{bmatrix}\right) = \begin{bmatrix} 2.45 & 1.63 & 4.08 \\ 1.63 & 1.09 & 2.72 \\ 1.22 & 2.45 & 3.67 \end{bmatrix}$$

#### 4.6 두 번째 레이어 계산
$$Z = \text{softmax}(\hat{A}H^{(1)}W^{(1)}) = \text{softmax}\left(\begin{bmatrix} 3.67 & 5.72 \\ 2.45 & 4.08 \\ 3.67 & 6.12 \end{bmatrix}\right)$$

---

## 5. 논리적 증명 또는 모델 특성 분석 (Proofs or Analysis)

### 5.1 계산 복잡도 분석

**시간 복잡도**: $O(|E|d^2)$
- 행렬 곱셈: $\hat{A}H^{(l)}W^{(l)}$
- $\hat{A}$는 sparse matrix (sparse-dense 곱셈)
- $|E|$: 엣지 수, $d$: feature 차원

**공간 복잡도**: $O(Nd + |E|)$
- 노드 feature: $O(Nd)$
- 인접 행렬: $O(|E|)$

### 5.2 Spectral 특성 분석

**명제**: GCN은 low-pass filter 역할을 한다

**증명**:
1. $\hat{A} = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$의 고유값은 $[-1, 1]$ 범위
2. 고주파 신호(고유값이 큰)는 감쇠
3. 저주파 신호(고유값이 작은)는 보존

### 5.3 Weisfeiler-Lehman 알고리즘과의 연결

**명제**: GCN은 WL 알고리즘의 neural generalization

**증명**:
1. WL 알고리즘: $h_i^{(l+1)} = \text{hash}(h_i^{(l)}, \{h_j^{(l)} : j \in \mathcal{N}_i\})$
2. GCN: $h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i||\mathcal{N}_j|}} W^{(l)}h_j^{(l)}\right)$
3. 두 방법 모두 이웃 정보를 집계하여 노드 표현 업데이트

### 5.4 수치적 안정성 증명

**명제**: Renormalization trick이 수치적 안정성을 보장

**증명**:
1. $\tilde{A} = A + I_N$으로 self-loop 추가
2. $\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$의 고유값이 $[0, 2]$ 범위
3. 고유값이 0이 되지 않아 수치적 안정성 확보

---

## 6. 실험 설정과 결과 요약 (Experiments & Results)

### 6.1 데이터셋 특성

| 데이터셋 | 노드 수 | 엣지 수 | 클래스 수 | Feature 수 | 라벨 비율 |
|---------|---------|---------|-----------|------------|-----------|
| Cora | 2,708 | 5,429 | 7 | 1,433 | 5.2% |
| Citeseer | 3,327 | 4,732 | 6 | 3,703 | 3.6% |
| Pubmed | 19,717 | 44,338 | 3 | 500 | 0.3% |
| NELL | 65,755 | 266,144 | 210 | 5,414 | 0.1% |

### 6.2 모델 설정

**하이퍼파라미터**:
- Learning rate: 0.01
- Dropout: 0.5
- L2 regularization: 5e-4
- Hidden units: 16 (NELL: 64)
- Optimizer: Adam
- Early stopping: 200 epochs

**모델 구조**:
- 2-layer GCN
- Layer 1: ReLU activation
- Layer 2: Softmax activation

### 6.3 Baseline 비교

**비교 대상**:
- Label Propagation
- SemiEmb
- ManiReg
- DeepWalk
- ICA
- Planetoid
- Chebyshev
- MLP

### 6.4 성능 결과

| 방법 | Cora | Citeseer | Pubmed | NELL |
|------|------|----------|--------|------|
| GCN | 81.5% | 70.3% | 79.0% | 66.0% |
| Planetoid* | 75.7% | 64.7% | 77.2% | 61.9% |
| DeepWalk | 70.7% | 51.4% | 74.3% | 58.1% |
| ICA | 75.1% | 69.1% | 73.9% | 66.0% |

**주요 성과**:
- 모든 데이터셋에서 최첨단 성능 달성
- 계산 효율성 향상 (wall-clock time 단축)
- Semi-supervised setting에서 우수한 성능

### 6.5 추가 실험 결과

**Karate Club 예제**:
- 34개 노드, 2개 클래스
- 4개 노드만 라벨 사용
- GCN이 클러스터 구조를 잘 포착

**Random Weight GCN**:
- 학습된 가중치 없이도 강력한 feature extractor
- 그래프 구조 정보만으로도 의미 있는 표현 학습

---

## 7. 한계점과 향후 방향 (Limitations & Future Directions)

### 7.1 현재 한계점

**계산적 한계**:
- Full-batch 학습으로 메모리 사용량 증가
- 대규모 그래프에서 mini-batch SGD 필요
- $O(|E|d^2)$ 계산 복잡도

**구조적 한계**:
- 1-hop 이웃만 고려 (고차 이웃 정보 손실)
- Undirected graph만 지원
- Edge feature 미사용
- Self-loop와 이웃 간 중요도 균형 문제

**실용적 한계**:
- Transductive setting에서만 검증
- Inductive setting에서 성능 저하
- 깊은 네트워크에서 over-smoothing 문제

### 7.2 향후 연구 방향

**확장성 개선**:
- Mini-batch 학습 방법 개발
- Sparse matrix 연산 최적화
- Graph partitioning 기반 병렬화

**기능 확장**:
- Directed graph 지원: $A_{ij} \neq A_{ji}$
- Edge feature 통합: $H^{(l+1)} = \sigma(\hat{A}H^{(l)}W^{(l)} + E^{(l)}W_E^{(l)})$
- 고차 이웃 attention: $H^{(l+1)} = \sigma(\sum_{k=1}^K \alpha_k \hat{A}^k H^{(l)}W^{(l)})$

**모델 개선**:
- Residual connection: $H^{(l+1)} = \sigma(\hat{A}H^{(l)}W^{(l)}) + H^{(l)}$
- Batch normalization: $H^{(l+1)} = \text{BN}(\sigma(\hat{A}H^{(l)}W^{(l)}))$
- Attention mechanism 도입

**해석 가능성**:
- Graph visualization 도구
- 중요 노드/엣지 식별
- Feature importance 분석

---

## 8. 한눈에 보는 요약 (TL;DR)

**GCN은 Chebyshev 다항식 근사와 renormalization trick을 통해 spectral graph convolution을 효율적으로 구현한 semi-supervised 노드 분류 모델이다. 핵심 수식 $H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})$로 1-hop 이웃 정보를 집계하며, $O(|E|d^2)$ 계산 복잡도로 기존 spectral 방법의 $O(N^3)$ 문제를 해결한다. Cora, Citeseer, Pubmed 데이터셋에서 최첨단 성능을 달성했으며, Weisfeiler-Lehman 알고리즘의 neural generalization으로 해석된다. 다만 transductive setting 제한, edge feature 미사용, over-smoothing 문제 등 개선 여지가 있다.**