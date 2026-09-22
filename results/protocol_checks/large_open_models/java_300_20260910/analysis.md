# Controlled Larger-Model Cross-Family Reliability Experiment Report

## 1. 개요 및 연구 목적
본 보고서는 FSE 2027 투고 논문("Do VLMs Judge Code Readability or Its Presentation? A Reliability Study of Pixels-Only Code Assessment")의 Figure 5(c)를 위한 **대형 오픈 VLM 3종 교차 계열 신뢰성 평가 실험**의 통계 분석 결과입니다.

- **데이터**: 기존 Figure 5 Qwen2.5-VL-32B 실험에서 사용된 정확히 동일한 300개 Java pair (`rq0_300_pair_pilot.csv`, easy/medium/hard 각 100쌍 균형)
- **모달리티**: `image-only` (두 snippet의 원본 코드 렌더링 이미지와 frozen A/B prompt만 제공, separate 2-image packaging)
- **평가 모델 3종**:
  1. `Qwen/Qwen2.5-VL-32B-Instruct` (BF16, greedy)
  2. `google/gemma-3-27b-it` (BF16, greedy)
  3. `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (BF16, greedy)

---

## 2. 모델별 핵심 평가 결과 (Aggregate Metrics)

| Model | Valid Acc. [95% CI] | Effective Acc. [95% CI] | Strict-Swap Error [95% CI] | Call Parse Failure | Pair Parse Failure |
|---|---:|---:|---:|---:|---:|
| **Qwen2.5-VL-32B-Instruct** | 0.673 [0.557, 0.782] | 0.507 [0.403, 0.609] | 0.247 [0.161, 0.341] | 0.0% | 0.0% |
| **gemma-3-27b-it** | 0.697 [0.550, 0.826] | 0.360 [0.259, 0.465] | 0.483 [0.380, 0.590] | 0.0% | 0.0% |
| **Mistral-Small-3.1-24B-Instruct-2503** | 0.610 [0.475, 0.739] | 0.370 [0.275, 0.472] | 0.393 [0.292, 0.496] | 0.0% | 0.0% |

> [!NOTE]
> - **Effective Accuracy**: Correct and valid pairs / All pairs (300 pairs)
> - **Valid Accuracy**: Correct and valid pairs / Strict-swap valid pairs
> - **Strict-swap Error**: 1 - (Strict-swap valid pairs / All pairs)
> - 모든 95% 신뢰구간은 273개 고유 snippet의 종속성을 보존하는 **10,000회 snippet-aware crossed cluster bootstrap**으로 계산되었습니다.

---

## 3. 모델 간 Paired 비교 및 가설 검정 (Paired Comparisons)

| Comparison | Effective Acc. Diff (pp) [95% CI] | Strict-Swap Error Diff (pp) [95% CI] | Exact McNemar (b/c) | Raw p-value | Holm-adjusted p-value |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-32B vs Gemma 3-27B | +14.67 pp [+5.00, +24.91] | -23.67 pp [-36.43, -10.56] | 103 / 32 | 6.7968e-10 | 2.0390e-09 |
| Qwen2.5-VL-32B vs Mistral Small 3.1-24B | +13.67 pp [+3.62, +24.15] | -14.67 pp [-27.96, -1.09] | 86 / 42 | 1.2536e-04 | 2.5072e-04 |
| Gemma 3-27B vs Mistral Small 3.1-24B | -1.00 pp [-11.90, +9.67] | +9.00 pp [-5.02, +23.13] | 52 / 79 | 0.0227 | 0.0227 |

---

## 4. 7대 핵심 연구 질문에 대한 실증적 분석 답변

1. **세 모델 중 어느 모델의 Effective Accuracy가 가장 높은가?**
   - **`Qwen2.5-VL-32B-Instruct`**가 **50.67%** (152/300)로 가장 높습니다. (Mistral: 37.00%, Gemma: 36.00%)

2. **그 차이의 cluster-bootstrap CI가 0을 제외하는가?**
   - **예, 완전히 제외하며 통계적으로 유의합니다.**
     - Qwen vs Gemma: 차이 **+14.67 %p** (95% CI: **[+5.00 %p, +24.91 %p]**, 0 불포함)
     - Qwen vs Mistral: 차이 **+13.67 %p** (95% CI: **[+3.62 %p, +24.15 %p]**, 0 불포함)
     - 반면 Gemma vs Mistral 차이(-1.00 %p, 95% CI: [-11.90 %p, +9.67 %p])는 0을 포함하여 차이가 유의하지 않습니다.

3. **어느 모델의 Strict-Swap Error가 가장 낮은가?**
   - **`Qwen2.5-VL-32B-Instruct`**가 **24.67%** (74/300, 95% CI: [16.11%, 34.09%])로 가장 낮습니다. (Mistral: 39.33%, Gemma: 48.33%)

4. **높은 Valid Accuracy가 낮은 validity 때문에 Effective Accuracy로 이어지지 않는 모델이 있는가?**
   - **예, `google/gemma-3-27b-it`가 완벽한 전형적 사례입니다.**
     - Gemma 3-27B의 Valid Accuracy는 **69.68%**로 3개 모델 중 **가장 높은 판정 정확도**를 보였으나,
     - 극심한 순서 불안정성(Strict-swap error **48.33%**, 유효 쌍 비율 단 51.67%)으로 인해 유효하지 않은 쌍이 대거 탈락하면서,
     - 최종 Effective Accuracy는 **36.00%**로 급락하여 3개 모델 중 최하위를 기록했습니다.
     - 이는 "순서 스왑 일관성을 검증하지 않고 Valid Accuracy만 보고하는 기존 평가 방식이 VLM의 신뢰성을 심각하게 왜곡할 수 있다"는 본 연구의 핵심 가설을 매우 강력하게 지지합니다.

5. **세 모델 모두에서 Order Instability(순서 불안정성)가 실질적으로 남아 있는가?**
   - **예, 세 모델 모두에서 상당한 수준의 Swap Error가 확인되었습니다.**
     - 가장 신뢰성이 높은 Qwen조차 4쌍 중 1쌍꼴인 **24.67%**에서 순서가 바뀌면 판정이 뒤집혔으며,
     - Mistral은 **39.33%**, Gemma는 거의 동전 던지기 수준인 **48.33%**의 순서 반전 오류를 기록했습니다.
     - 따라서 24B–32B급의 대형 VLM으로 스케일업되더라도 스크린샷 픽셀 기반 코드 판정의 순서 민감성은 완전히 해소되지 않고 실질적으로 상존합니다.

6. **Parse Failure가 결과 차이를 설명하는가?**
   - **아닙니다.** 본 통제 실험에서는 공식 Chat Template의 System Instruction 설정을 통해 **세 모델 모두 Call-level 및 Pair-level Parse Failure Rate 0.0% (0/600 calls, 0/300 pairs 전수 성공)**를 달성했습니다.
   - 따라서 관찰된 Effective Accuracy와 Strict-swap Error의 격차는 파싱 실패로 인한 잡음이 아니라, **순수한 모델 고유의 순서 신뢰성 및 코드 판정 능력 차이**에서 기인합니다.

7. **기존 Figure 5의 결론을 강화하는가, 약화하는가, 또는 혼합된 결과인가?**
   - **기존 Figure 5의 핵심 결론을 매우 강력하게 강화(Strengthen)합니다.**
     - 기존 Figure 5(c)는 Qwen32B(text+image 300 Java)와 Qwen122B(image-only 300 multi-lang)의 이종 조건 비교라는 한계가 있었습니다.
     - 본 통제 실험은 정확히 동일한 300개 Java snippet pairs, 동일한 image-only 모달리티, 동일한 BF16 Greedy 디코딩 환경 하에서 서로 다른 3대 계열의 대형 모델을 정밀 비교함으로써,
     - Figure 5(c)를 **완벽히 통제된 matched larger-model cross-family reliability check**로 성공적으로 교체할 수 있음을 입증합니다.

---

## 5. 논문 작성 지침 및 허용/금지 주장

### 허용 가능한 주장 (Allowed Claims):
- "Under an exact matched protocol on 300 Java snippet pairs (image-only, BF16 greedy, separate 2-image packaging), substantial order instability (strict-swap error: 24.7%–48.3%) persists across three 24B–32B open VLMs from distinct families."
- "Gemma 3-27B achieves the highest valid accuracy (69.7%) but drops to 36.0% effective accuracy due to a 48.3% swap error, demonstrating that validity filtering is indispensable."
- "Order instability is not an artifact of small model sizes (e.g., 7B) but remains a persistent reliability challenge in 24B–32B scale open VLMs."

### 금지되는 주장 (Disallowed Claims):
- ❌ "Larger models are universally less reliable than smaller models." (인과적 스케일링 효과 주장 금지)
- ❌ "Increasing model size does not improve reliability." (크기 증가의 인과성 부정 금지)
- ❌ "24B–32B models represent all frontier VLMs." (최첨단 프론티어 전체로의 무리한 일반화 금지)
- ❌ "The experiment establishes a scaling law." (스케일링 법칙 확립 주장 금지)

---

## 5. 산출물 파일 목록 및 위치
- Experiment Manifest: `experiment_manifest.json`
- Pair IDs & Ground Truth: `pair_ids.csv`
- Raw Outputs: `qwen25_vl_32b_raw.jsonl`, `gemma3_27b_raw.jsonl`, `mistral_small_31_24b_raw.jsonl`
- Pair-level Results: `pair_level_results.csv`
- Aggregate Metrics: `aggregate_metrics.csv`
- Paired Comparisons: `paired_comparisons.csv`
