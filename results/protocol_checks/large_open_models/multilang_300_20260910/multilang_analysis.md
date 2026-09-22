# 다국어(Java · Python · CUDA) 제어 기반 대형 오픈 VLM 가독성 판단 신뢰성 분석 보고서

- **실험 일시**: 2026년 9월 10일 (KST)
- **대상 논문**: FSE 2027 투고 논문 (*"Do VLMs Judge Code Readability or Its Presentation? A Reliability Study of Pixels-Only Code Assessment"*)
- **실험 범위**: Java(300쌍) + Python(300쌍) + CUDA(300쌍) = **총 900쌍 (1,800 unique AB/BA calls / model, 총 5,400 calls)**
- **평가 모델**:
  1. `Qwen/Qwen2.5-VL-32B-Instruct` (rev: `7cfb30d71a1f4f49a57592323337a4a4727301da`)
  2. `google/gemma-3-27b-it` (rev: `005ad3404e59d6023443cb575daa05336842228a`)
  3. `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (rev: `68faf511d618ef198fef186659617cfd2eb8e33a`)
- **실험 환경**: 단일 NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB VRAM), BF16 greedy (`do_sample=False`, `temperature=None`), `max_new_tokens=24`, `batch_size=1`, seed 42.

---

## 1. 종합 실험 결과 요약

### 1.1 전체(900쌍, Java + Python + CUDA) 통합 지표 (10,000회 Cluster Bootstrap 95% CI)

| 모델명 | 총 페어 수 | 파싱 실패율 | 순서 불일치율 (Swap Error) [95% CI] | 유효 정확도 (Valid Acc) [95% CI] | **유효 일치 정확도 (Effective Acc)** [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-VL-32B-Instruct** | 900 | **0.0%** (0/1800) | **21.11%** [17.54%, 24.73%] | **72.96%** [68.38%, 77.40%] | **57.56%** [53.09%, 61.96%] |
| **Mistral-Small-3.1-24B-Instruct** | 900 | **0.0%** (0/1800) | **48.00%** [43.06%, 52.74%] | **65.60%** [59.56%, 71.59%] | **34.11%** [29.85%, 38.42%] |
| **Gemma-3-27B-it** | 900 | **0.0%** (0/1800) | **48.00%** [43.16%, 52.86%] | **62.82%** [56.52%, 69.19%] | **32.67%** [28.28%, 37.08%] |

> **수학적 항등식 검증 완수**: $\text{Effective Acc} = \text{Valid Acc} \times (1 - \text{Swap Error})$
> - Qwen: $72.96\% \times (1 - 0.2111) = 57.56\%$ ✅
> - Mistral: $65.60\% \times (1 - 0.4800) = 34.11\%$ ✅
> - Gemma: $62.82\% \times (1 - 0.4800) = 32.67\%$ ✅

---

### 1.2 언어별(Java vs Python vs CUDA) 세부 비교 (각 언어별 300쌍)

#### A. Java (300 pairs: Easy 100 / Med 100 / Hard 100)
| 모델명 | Strict-Swap Error | Valid Accuracy | **Effective Accuracy** [95% CI] |
| :--- | :---: | :---: | :---: |
| **Qwen2.5-VL-32B** | **24.67%** | 67.26% | **50.67%** [42.98%, 58.33%] |
| **Mistral-Small-3.1-24B** | 39.33% | 60.99% | **37.00%** [30.00%, 43.88%] |
| **Gemma-3-27B** | 48.33% | **69.68%** | **36.00%** [28.44%, 43.59%] |

#### B. Python (300 pairs: Easy 100 / Med 100 / Hard 100)
| 모델명 | Strict-Swap Error | Valid Accuracy | **Effective Accuracy** [95% CI] |
| :--- | :---: | :---: | :---: |
| **Qwen2.5-VL-32B** | **20.00%** | **73.33%** | **58.67%** [51.35%, 65.75%] |
| **Gemma-3-27B** | 43.67% | 52.66% | **29.67%** [21.95%, 37.30%] |
| **Mistral-Small-3.1-24B** | **65.00%** ⚠️ | 63.81% | **22.33%** [16.48%, 28.46%] |

#### C. CUDA (300 pairs: Easy 100 / Med 100 / Hard 100)
| 모델명 | Strict-Swap Error | Valid Accuracy | **Effective Accuracy** [95% CI] |
| :--- | :---: | :---: | :---: |
| **Qwen2.5-VL-32B** | **18.67%** | **77.87%** | **63.33%** [55.64%, 71.43%] |
| **Mistral-Small-3.1-24B** | 39.67% | 71.27% | **43.00%** [35.11%, 50.89%] |
| **Gemma-3-27B** | 52.00% ⚠️ | 67.36% | **32.33%** [24.77%, 40.15%] |

---

## 2. 모델 간 쌍대 비교 (Exact McNemar Test)

동일한 900쌍에 대해 두 모델의 유효 일관성(Strict-Swap 일관 여부) 전이 행렬을 구성하여 McNemar 검정을 수행함:

| 비교군 | 언어 | 페어 수 | Effective Acc 차이 | Swap Error 차이 | McNemar (b / c) | Exact p-value | 통계적 유의성 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen 32B vs Gemma 27B** | **ALL** | 900 | **+24.89%p** | **-26.89%p** | 327 / 85 | **$1.49 \times 10^{-34}$** | 극단적 유의 ($p < 0.001$) |
| Qwen 32B vs Gemma 27B | Java | 300 | +14.67%p | -23.67%p | 103 / 32 | $6.80 \times 10^{-10}$ | 매우 유의 ($p < 0.001$) |
| Qwen 32B vs Gemma 27B | Python | 300 | +29.00%p | -23.67%p | 99 / 28 | $1.72 \times 10^{-10}$ | 매우 유의 ($p < 0.001$) |
| Qwen 32B vs Gemma 27B | CUDA | 300 | +31.00%p | -33.33%p | 125 / 25 | $3.41 \times 10^{-17}$ | 극단적 유의 ($p < 0.001$) |
| **Qwen 32B vs Mistral 24B** | **ALL** | 900 | **+23.44%p** | **-26.89%p** | 332 / 90 | **$1.15 \times 10^{-33}$** | 극단적 유의 ($p < 0.001$) |
| Qwen 32B vs Mistral 24B | Java | 300 | +13.67%p | -14.67%p | 86 / 42 | $0.000125$ | 매우 유의 ($p < 0.001$) |
| Qwen 32B vs Mistral 24B | Python | 300 | **+36.33%p** | **-45.00%p** | 157 / 22 | **$2.56 \times 10^{-26}$** | 극단적 유의 ($p < 0.001$) |
| Qwen 32B vs Mistral 24B | CUDA | 300 | +20.33%p | -21.00%p | 89 / 26 | $2.95 \times 10^{-09}$ | 매우 유의 ($p < 0.001$) |
| **Gemma 27B vs Mistral 24B** | **ALL** | 900 | -1.44%p | 0.00%p | 190 / 190 | 1.000 | 유의차 없음 ($p = 1.0$) |
| Gemma 27B vs Mistral 24B | Java | 300 | -1.00%p | +9.00%p | 52 / 79 | 0.0227 | 유의 ($p < 0.05$) |
| Gemma 27B vs Mistral 24B | Python | 300 | +7.33%p | -21.33%p | 99 / 35 | $2.89 \times 10^{-08}$ | 매우 유의 ($p < 0.001$) |
| Gemma 27B vs Mistral 24B | CUDA | 300 | -10.67%p | +12.33%p | 39 / 76 | $0.000717$ | 매우 유의 ($p < 0.001$) |

---

## 3. 언어 구문론적(Syntax/Presentation) 특성에 따른 핵심 발견

### 발견 1: Python의 공백/들여쓰기(Whitespace-sensitive) 구조와 Mistral의 순서 반전 붕괴
- **현상**: Mistral-Small-3.1-24B는 Java(39.3%)나 CUDA(39.7%)에서는 30%대 Swap error를 유지했으나, **Python에서는 Swap error가 무려 65.0%로 폭증**했습니다. 유효한 페어가 300쌍 중 단 105쌍에 불과했으며, Effective Accuracy는 **22.33%**로 추락했습니다.
- **원인 분석**: Python은 중괄호 `{}` 없이 탭/공백 들여쓰기로 블록 구조가 결정됩니다. 픽셀-온리(Pixel-only) 환경에서 2개 이미지가 제시될 때, Pixtral 비전 인코더의 2D 패치 위치 임베딩이 입력 순서(첫 번째 이미지 vs 두 번째 이미지)에 따라 들여쓰기 공간 구조를 다르게 편향되게 인식함을 시사합니다. 반면 Qwen2.5-VL-32B는 NaViT 기반 동적 해상도 인코딩을 사용하여 Python에서도 20.0%의 낮은 Swap error를 보였습니다.

### 발견 2: CUDA의 높은 기호 밀도(Syntax-dense)와 Qwen의 탁월한 강인성
- **현상**: CUDA 커널 코드는 포인터 연산(`*`, `->`), 템플릿, 스레드 인덱스 매크로(`threadIdx.x`, `blockIdx.x`) 등 텍스트 밀도가 매우 높습니다.
- **결과**: 놀랍게도 Qwen2.5-VL-32B는 CUDA에서 가장 높은 **Effective Accuracy 63.33%**, **Valid Accuracy 77.87%**, **Swap Error 18.67%**를 기록했습니다. Mistral 역시 CUDA에서 Effective Accuracy 43.0%로 Java/Python보다 우수했습니다.
- **해석**: 가독성 차이가 구조적 타이포그래피나 공백보다는 명확한 식별자 명명 규칙 및 연산자 배치에 좌우되는 CUDA 코드에서 VLM 비전 인코더가 시각적 단서를 더 명확하게 포착함을 보여줍니다.

### 발견 3: Gemma-3-27B의 "위장된 정확도(Phantom Accuracy)"
- **현상**: Gemma-3-27B는 개별 판정 시 유효 정확도(Valid Acc)가 Java에서 69.68%, CUDA에서 67.36%로 높은 편입니다.
- **치명적 결함**: 그러나 순서가 바뀌면 정반대로 선택을 뒤집는 **Swap error가 전체 48.0%, CUDA 52.0%**에 달해, 유효 일관성을 충족하는 Effective Accuracy는 32.67%에 머물렀습니다. 이는 단방향(AB만 평가)으로 평가할 경우 모델의 가독성 판단 능력이 실제보다 최대 2배 이상 과대평가될 수 있음을 증명하는 강력한 실증적 증거입니다.

---

## 4. 논문(FSE 2027) 기여 및 학술적 시사점

1. **Figure 5(c) 완벽 대체 및 일반화 증명**:
   - 24B–32B급 3대 오픈 모델 패밀리(Qwen, Gemma, Mistral)에 걸쳐, 단일 언어(Java)에 국한되지 않고 동적 인터프리터 언어(Python)와 고성능 병렬 시스템 언어(CUDA) 전반에서 순서 편향(Order Sensitivity)이 편재함을 입증하였습니다.
2. **파싱 실패율 0.0%의 무결성 통제**:
   - 3개 모델 모두 `max_new_tokens=24` 예산 내에서 1,800 calls 전수 0건의 파싱 실패(총 5,400 calls 중 0 실패)를 달성하여, 불완전 파싱으로 인한 지표 왜곡을 100% 제거했습니다.
3. **통계적 엄밀성**:
   - 10,000회 Crossed Cluster Bootstrap 신뢰구간과 Exact McNemar Test를 통해, Qwen2.5-VL-32B가 Gemma 및 Mistral 대비 통계적으로 유의미하게($p < 10^{-33}$) 높은 순서 신뢰성을 보임을 확정했습니다.

---

## 5. 10대 무결성 품질 보증(QA) 통과 기록

1. ✅ 3개 모델 각 언어별 600 calls (총 1,800 calls/model, 3개 모델 합계 5,400 calls) 전수 일치
2. ✅ 900개 기준 페어 ID(Java 300, Python 300, CUDA 300) 100% 1:1 매핑
3. ✅ 모든 페어당 정확히 1개 AB, 1개 BA 순서 매핑
4. ✅ 중복 키(Duplicate Key) 0건
5. ✅ 페어 레벨 2,700개 행(3 모델 × 900 페어) 완벽 생성
6. ✅ 9개 셀(3개 언어 × 3개 난이도) 각 100쌍씩 균등 배치 확인
7. ✅ 수학적 항등식($\text{Effective Acc} = \text{Valid Acc} \times \text{Valid Rate}$) 오차 0.000%
8. ✅ 파싱 실패 페어의 유효 플래그 오염 0건
9. ✅ 원본 이미지 508개 SHA-256 해시 검증 완료
10. ✅ Deterministic BF16 Greedy (`temperature=None`, `do_sample=False`) 재현성 보장
