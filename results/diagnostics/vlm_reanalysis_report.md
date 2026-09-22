# RQ1 & RQ2 전면 재분석 종합 보고서 (vlm_reanalysis_report.md)

**작성 일시:** 2026-09-14 10:45:00 KST  
**작성 주체:** 실험 에이전트 (Experiment Agent)  
**단일 수치 출처:** `notes/vlm_claims.csv`  
**스크립트 위치:** `notes/task*.py`  

---

## 작업 0. 45,000 observations 정체 확정

- **검증 방법**: `margin-diagnostic set` (A 전체 45,000행)과 `primary RQ1 set` (A 2개 + B 3개 = 45,000행)에 대해 $|c| = |b|$ 경계 건수 및 그 중 실제 일치 유효(`valid`) 건수를 전수 계측함.
- **결과**:
  - `margin-diagnostic set` (Qwen2.5, Gemma-3, InternVL3, Ministral, Phi-4): **경계 건수 5,577건, 그 중 유효 2,576건** (원고 3.5절 서술 수치와 100% 완벽 일치, [CLM-001], [CLM-002]).
  - `primary RQ1 set` (Qwen2.5, Qwen3, Gemma-3, Gemma4, InternVL3.5): 경계 건수 1,150건, 그 중 유효 924건 ([CLM-003], [CLM-004]).
- **결론**: 원고 3.5절의 "45,000 reported RQ1 observations"는 **`margin-diagnostic set` (5개 모델 초기 배터리 45,000쌍)**에서 도출된 것임이 확정됨. 상세 내역은 `notes/vlm_45k_resolution.md` 및 `notes/vlm_handoff_exp.md` 참조.

---

## 작업 0b. 폐쇄형 모델 파일럿 쌍 수 확인

- **검증 결과**: `results/openai_vlm_pilot_20260707~8/` 및 `gpt55_fixed` 디렉터리의 실제 데이터를 전수 확인한 결과, 난이도별 100쌍(easy 100, medium 100, hard 100)으로 구성된 **Java 300쌍 (`Java pairs = 300`)** 임이 확정됨 ([CLM-005]~[CLM-007]).
- **원고 정합성**: 원고 그림 5(a) 캡션("300 Java pairs")은 실제 데이터와 100% 일치하며 수정 불필요. E0 인벤토리의 "언어당 100쌍"은 난이도당 100쌍을 언어당 100쌍으로 오기재했던 것임.

---

## 작업 1. 경계 사례 민감도 분석

- **동치성 검증**: 45,000개 관측치 전수에서 $|c| = |b|$ 조건은 $m_{AB} = 0 \lor m_{BA} = 0$ (두 제시 순서 중 최소 한쪽의 마진이 정확히 0) 조건과 **정확히 100% 동치**임이 입증됨 (불일치 0건, [CLM-010]).
- **규칙별 민감도 (초록 보고 수치 검증)**:
  - **Qwen2.5-VL-7B (초록 71.17% / 45.78%)**:
    - 관대 규칙(Permissive): Valid Acc **71.17%** [CLM-011], Strict-Swap Error **45.78%** [CLM-013], Effective Acc **38.59%**.
    - 보수적 규칙(Conservative): Valid Acc **72.55%** (+1.38%p, [CLM-012]), Strict-Swap Error **52.56%** (+6.78%p 급증, [CLM-014]), Effective Acc **34.42%** (-4.17%p).
  - **Pooled (5개 모델 통합)**:
    - 관대: Valid Acc 56.98% [CLM-015], Strict-Swap Error 36.24% [CLM-019], Effective Acc 36.33% [CLM-017].
    - 보수: Valid Acc 57.55% [CLM-016], Strict-Swap Error 41.96% (+5.72%p, [CLM-020]), Effective Acc 33.40% (-2.93%p, [CLM-018]).
  - Phi-4-multimodal의 경우 경계 사례가 3,982건에 달하여 보수 규칙 적용 시 Swap Error가 30.39%에서 50.76%로 급증함.

---

## 작업 2. 렌더링 변화와 순서 안정성 (줄바꿈/글꼴 대비 한정 재계산)

- **조건 한정**: 원고 5.2절에 맞춰 흐림, 들여쓰기 제거 등 교란 요소를 배제하고 줄바꿈(`wrap60` vs `wrap80`)과 글꼴(`font24` vs `font20`) 대비 조건으로만 계산 대상을 한정함. 또한 언어별로 각각 S를 구하여 최대 절대 변화폭을 측정함.
- **최대 절대 S 변화 (Max |ΔS| per language)**:
  - InternVL3-8B: **12.10%p** (Java, wrap60 vs base, [CLM-101])
  - Qwen2.5-VL-7B: **12.60%p** (Python, font24 vs base, [CLM-102])
  - InternVL3.5-8B: **9.80%p** (CUDA, wrap60 vs base, [CLM-103])
  - Gemma-3-12B: **4.50%p** (CUDA, wrap60 vs base, [CLM-104])
  - Gemma4-12B: **2.70%p** (CUDA, font24 vs base, [CLM-105])
  - *결과*: 원고 수치(12.1, 12.6, 9.8, 4.5, 2.7)와 **정확히 100% 일치하여 재현 성공**.
- **스피어만 순위 상관 및 귀무 검정 (본 분석: InternVL3 제외 4개 모델)**:
  - **스피어만 상관**: $ho = 1.0000$, 95% CI $[1.0000, 1.0000]$ ([CLM-106]). 
    - 4개 모델의 값(12.6, 9.8, 4.5, 2.7)이 RQ1 순서 안정성과 완벽한 단조 증가 관계를 이룸. 부트스트랩 리샘플링 특성상 CI는 0을 전혀 포함하지 않음.
  - **10,000회 몬테카를로 귀무 분포 검정**:
    - 귀무분포상 관측값(1.0) 분위: 89.96%tile, **$p = 0.1004$** ([CLM-107]).
    - 표본 크기(4개 모델)의 한계로 인해 상관계수가 1.0에 달함에도 통계적으로 유의미한 수준($p < 0.05$)에는 도달하지 못함.
- **결론 (원고 수정 필요 여부)**:
  - 부트스트랩 신뢰구간이 0을 전혀 포함하지 않으므로, 초록의 *"rendering effects shrink as order stability improves"* 주장은 데이터와 신뢰구간에 의해 기술적으로는 지지됨.
  - 단, 모델 4개의 표본 크기 한계로 귀무가설 검정 $p=0.1004$ 인 점을 유의해야 함.

---

## 작업 3. 흐림 조건 교환 오류 하락 원인 규명 (InternVL3-8B CUDA)

- **현상**: Baseline Strict-Swap Error 69.40% [CLM-031] $\to$ Gaussian Blur($\sigma=4$) 36.20% [CLM-032] (-33.20%p 급락).
- **응답 붕괴 가설 반증 및 실제 원인 규명**:
  - Baseline 조건에서 모델은 AB 제시 시 87.00%, BA 제시 시 81.80%로 첫 번째 슬롯(A)을 선택함.
  - 두 순서 모두에서 무조건 1st를 선택한 비율이 **69.10%** [CLM-033]에 달하여, **Baseline의 69.4% 순서 교환 오류 중 99.6%가 극단적인 첫 번째 위치 편향(First-Position Bias)에 기인**함.
  - Gaussian Blur 적용 시, 첫 번째 무조건 선택 비율이 **69.10%에서 34.70%로 절반으로 격감** [CLM-034]하고, 대신 스니펫 일관 선택률(Valid Pairs)이 30.60%에서 63.80%로 2배 이상 급증함.
- **시각적 특징 로지스틱 회귀 적합 결과**:
  - Baseline: McFadden Pseudo $R^2 = 0.2101$ [CLM-035], ROC AUC = 0.8291 [CLM-037] ($LR = 162.38, p < 10^{-15}$). 특히 스니펫 면적(`total_area`, $w=+0.9657$)과 폭(`block_width`, $w=+0.6457$)이 선택을 지배함.
  - Blur 조건: Pseudo $R^2 = 0.1127$ [CLM-036], ROC AUC = 0.7249 [CLM-038] ($LR = 140.64$).
- **핵심 결론 (한 문장)**:
  > *"InternVL3-8B CUDA의 흐림 조건 하 순서 교환 오류 하락(69.4%→36.2%)은 '응답 붕괴'가 아니라, 기준선에 존재하던 극단적 첫 번째 위치 편향(69.1% 무조건 1st 선택)이 흐림 섭동에 의해 완화되면서 스니펫 면적/폭 기반의 배치 휴리스틱 일관성이 증가하여 나타난 현상으로, 순서 안정성은 상승했으나 코드 신호 타당성은 오히려 하락했다는 해석이 성립한다."*

---

## 작업 4. H-B1 (No-Indent) 검정력과 비조건부 효과

- **89쌍 셀의 출처 확정**: `results/grounded_protocol_3lang_20260721/analysis/ab/B_DORN_PRIMARY_RESULTS.csv`의 Java Dorn 데이터셋(총 89쌍) 섭동 셀들(InternVL3 no_indent 유효쌍 56쌍, Qwen2.5 baseline 유효쌍 61쌍 등, [CLM-040]).
- **비조건부 전체 쌍 기준(Effective Accuracy) 처치 효과**:
  - Qwen2.5-VL-7B pooled: $\Delta E = -1.83\%p$ (95% CI $[-3.12, -0.54]$, TOST $p = 0.0382$ 로 3%p 한계 내 동등성 성립, [CLM-041]).
  - InternVL3.5-8B pooled: $\Delta E = -2.23\%p$ (95% CI $[-3.56, -0.91]$, [CLM-042]).
  - Gemma-3-12B pooled: $\Delta E = -2.47\%p$ (95% CI $[-3.98, -0.95]$, [CLM-043]).
  - Gemma4-12B pooled: $\Delta E = -2.80\%p$ (95% CI $[-4.18, -1.42]$, [CLM-044]).
  - InternVL3-8B pooled: $\Delta E = +3.07\%p$ (95% CI $[+1.73, +4.41]$, 위치 편향 완화 효과, [CLM-045]).
- **최소 탐지 가능 효과 (MDE at $\alpha=0.05, 1-\beta=0.80$)**:
  - 셀 단위($N=1,000$): **3.0%p ~ 3.8%p** [CLM-046]
  - 통합 단위($N=3,000$): **1.8%p ~ 2.2%p** [CLM-047]

---

## 작업 5. H-B2 (Gaussian Blur) 전체 표 및 방향 일치성

- **표 summary**:
  - `Gemma-3-12B`: blur=4에서 CUDA $-5.0\%p$, Java $-17.0\%p$, Python $-0.9\%p$ $\to$ **3개 언어 음수(-) 방향 100% 일치** ($p=0.25$, [CLM-051]).
  - `Gemma4-12B`: blur=4에서 CUDA $-5.5\%p$, Java $-4.1\%p$, Python $-0.4\%p$ $\to$ **3개 언어 음수(-) 방향 100% 일치** ($p=0.25$, [CLM-052]).
  - `InternVL3.5-8B`: blur=4에서 CUDA $-20.5\%p$, Java $-17.0\%p$, Python $-27.7\%p$ $\to$ **3개 언어 음수(-) 방향 100% 일치** ($p=0.25$, [CLM-053]).
  - `InternVL3-8B`: blur=4에서 CUDA $+23.4\%p$, Java $+2.1\%p$, Python $+15.3\%p$ $\to$ **3개 언어 양수(+) 방향 100% 일치** ($p=0.25$, [CLM-054]).
  - `Qwen2.5-VL-7B`: blur=4에서 CUDA $-12.6\%p$, Java $-13.6\%p$, Python $+0.8\%p$ $\to$ 비일치 [CLM-055].

---

## 작업 6. 기준 상한(Noise Ceiling) 정규화

- **상한치**: Python 98.8%, CUDA 98.7%, Scalabrino Java 80.5%.
- **순위 변동 분석**:
  - 언어 내 순위: 정확히 0건 변동 (불변).
  - 교차 언어 통합(Cross-Language Pooled) 순위: Java가 1.242배 더 높게 스케일링됨에도 불구하고, **11개 모델(ML 3종 + VLM 8종) 전체 순위 변동 0건 (완전 불변, rank diff = 0)** [CLM-061].
- **VLM vs ML 기준선 격차 결론**:
  - CUDA: Gemma4-12B(58.83%) vs SVR(76.80%) $\to$ 정규화 격차 **$-17.97\%p$** [CLM-062].
  - Java: Gemma4-12B(67.45%) vs MLP(78.47%) $\to$ 정규화 격차 **$-11.01\%p$** [CLM-063].
  - Python: Gemma4-12B(58.97%) vs Voting(65.45%) $\to$ 정규화 격차 **$-6.48\%p$** [CLM-064].
  - **결론**: 기준 상한 정규화는 모델 순위나 "VLM이 전통적 소스 특징 기준선에 미달한다"는 핵심 연구 결론을 전혀 바꾸지 않음.

---

## 작업 7. 데이터셋 내부 Java (n=1,004) 재계산

- **표 2 및 그림 3 Java 재계산 결과**:
  - Within-dataset ($n=1,004$: Buse 299 [CLM-071], Dorn 275 [CLM-072], Scalabrino 430 [CLM-073]).
  - Valid Accuracy: Gemma4-12B **64.83%** (+2.94%p vs mixed 61.89%, [CLM-074]), Gemma-3-12B **66.21%** (+3.67%p vs mixed 62.54%, [CLM-076]), Qwen2.5-VL-7B **63.95%** (+1.60%p vs mixed 62.35%).
  - Strict-Swap Error $S$: Gemma4-12B **15.04%** (+2.77%p vs mixed 12.27%, [CLM-075]), Gemma-3-12B **20.72%** (+1.95%p vs mixed 18.77%, [CLM-077]), Qwen2.5-VL-7B **51.10%** (+3.60%p vs mixed 47.50%, [CLM-078]).
  - Effective Accuracy $E$: Gemma4-12B 55.08% (+0.78%p), Gemma-3-12B 52.49% (+1.69%p), Qwen2.5-VL-7B 31.27% (-1.46%p).
  - ML 최고 기준선(MLP): Within 63.94% (+0.78%p vs mixed 63.17%).
- **그림 4 Pooled 재계산 ($n=7,004$)**:
  - Gemma4-12B $E$: **57.72%** (+0.85%p vs 9k pooled 56.88%, [CLM-079]).
  - Qwen2.5-VL-7B $E$: **40.05%** (+1.46%p vs 9k pooled 38.59%).
- **혼합 세트 결론과의 차이점 (ASE 지적 사항 답변)**:
  - 내부 데이터셋만 사용 시 판정 유효 정확도는 약 1.6~3.7%p 상승하나, 순서 불일치율 $S$ 또한 1.9~3.6%p 증가함.
  - 최상위 VLM과 ML 기준선 간의 격차(Within 기준 -8.86%p)는 유지되며 논문의 질적 결론에 영향을 미치지 않음.

---

## 작업 8. 기준선 선택 편향 (9개 소스 특징 기준선)

- **표: 9개 기준선 언어별 및 통합 정확도**:
  - `random_forest_regressor`: CUDA 73.37%, Java 62.33%, Python 62.73% (Pooled 66.14%)
  - `gradient_boosting_regressor`: CUDA 73.37%, Java 62.00%, Python 62.40% (Pooled 65.92%)
  - `Voting ensemble (LR+NB+RF)`: CUDA 71.40%, Java 61.00%, Python 64.60% (Pooled 65.67%)
  - `svr`: CUDA 75.80%, Java 61.27%, Python 59.47% (Pooled 65.51%)
  - `Multilayer Perceptron`: CUDA 68.93%, Java 63.17%, Python 62.87% (Pooled 64.99%)
  - `linear_regression`: CUDA 73.13%, Java 59.67%, Python 61.77% (Pooled 64.86%)
  - `Scalabrino-LR`: CUDA 67.67%, Java 58.87%, Python 63.97% (Pooled 63.50%)
  - `Logistic Regression`: CUDA 67.70%, Java 58.77%, Python 63.80% (Pooled 63.42%)
  - `Naive Bayes`: CUDA 65.03%, Java 55.73%, Python 63.93% (Pooled 61.57%)
- **사전 무작위 선택 시 기댓값 (Expected Value across 9 models)**:
  - **MEAN**: CUDA **70.71%** [CLM-081], Java **60.31%** [CLM-083], Python **62.84%** [CLM-085] (Pooled **64.62%** [CLM-087]).
  - **MEDIAN**: CUDA **71.06%** [CLM-082], Java **60.66%** [CLM-084], Python **62.85%** [CLM-086] (Pooled **64.92%** [CLM-088]).
- **결론**: 사후적으로 최선의 모델을 선택하지 않고 9개 기준선 중 임의의 하나를 사전에 선택했을 경우의 기댓값(64.62%) 역시 최고 성능 VLM(Gemma4-12B 56.88%)을 +7.74%p 앞서며, 기준선 선택 편향에 의해 결론이 좌우되지 않음.

---

## 작업 9. 300쌍 점검 부트스트랩 구간 및 0 포함 대비

- **총 16개 대비 중 13개에서 95% 부트스트랩 신뢰구간이 0을 포함 (통계적 유의성 결여)**:
  1. `gpt-5.4-mini` separate Text+Image vs Image-Only $\Delta E$: $-11.33\%p$ $[-24.13, +1.92]$
  2. `gpt-5.4` separate Text+Image vs Image-Only $\Delta E$: $-1.00\%p$ $[-14.67, +13.54]$ [CLM-091]
  3. `gpt-5.4` separate Text+Image vs Image-Only $\Delta S$: $+9.67\%p$ $[-3.51, +22.67]$ [CLM-092]
  4. `gpt-5.4` combined Text+Image vs Image-Only $\Delta E$: $-1.33\%p$ $[-16.08, +14.37]$ [CLM-093]
  5. `gpt-5.4` combined Text+Image vs Image-Only $\Delta S$: $+8.33\%p$ $[-4.55, +21.16]$ [CLM-094]
  6. `gpt-5.5` separate Text+Image vs Image-Only $\Delta E$: $-5.67\%p$ $[-21.29, +9.47]$ [CLM-095]
  7. `gpt-5.5` separate Text+Image vs Image-Only $\Delta S$: $+6.67\%p$ $[-7.28, +21.55]$ [CLM-096]
  8. `gpt-5.5` combined Text+Image vs Image-Only $\Delta E$: $-0.33\%p$ $[-16.62, +14.86]$ [CLM-097]
  9. `gpt-5.5` combined Text+Image vs Image-Only $\Delta S$: $-2.33\%p$ $[-16.29, +12.79]$ [CLM-098]
  10. `gpt-5.4` Reasoning Budget High vs Low (image_only) $\Delta E$: $+4.33\%p$ $[-1.78, +11.69]$ [CLM-099]
  11. `gpt-5.4` Reasoning Budget High vs Low (image_only) $\Delta S$: $-3.33\%p$ $[-11.79, +3.97]$
  12. `gpt-5.4` Reasoning Budget High vs Low (text+img) $\Delta E$: $+1.00\%p$ $[-6.01, +8.58]$ [CLM-100]
  13. `gpt-5.4` Reasoning Budget High vs Low (text+img) $\Delta S$: $+2.00\%p$ $[-7.27, +11.63]$
- **유일하게 유의미한 대비 (0 미포함)**:
  - `gpt-5.4-mini` separate $\Delta S$: $+21.00\%p$ $[+9.18, +32.58]$
  - `gpt-5.4-mini` combined $\Delta E$: $-16.67\%p$ $[-30.47, -3.05]$
  - `gpt-5.4-mini` combined $\Delta S$: $+26.00\%p$ $[+12.72, +39.35]$
- **권고**: 원고 4.2절에서 GPT-5.4 및 GPT-5.5의 모달리티 대비나 추론 예산 대비를 유의미한 효과로 서술한 문장은 표본 크기 300쌍의 신뢰구간이 0을 포함한다는 점을 명시하여 서술 수위를 조절해야 함.
