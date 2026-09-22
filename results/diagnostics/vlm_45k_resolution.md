# 45,000 관측치 정체 확정 보고서 (vlm_45k_resolution.md)

**작성 일시:** 2026-09-14 10:40:00 KST  
**작성 주체:** 실험 에이전트 (Experiment Agent)  
**참조 원고:** FSE 2027 - VLM-as-a-Judge 원고 3.5절 및 4.2절

---

## 1. 분석 배경 및 질문

원고 3.5절은 다음과 같이 기술하고 있습니다:
> *"Across all 45,000 reported RQ1 observations, 5,577 cases lie on the boundary $|c| = |b|$, of which 2,576 are actually valid under two-order presentation."*

그러나 원고의 분석 집합 중 행 수가 45,000행인 집합이 두 개 존재하여 출처 집합의 특정이 필요했습니다:
1. **`primary RQ1 set`**: Qwen2.5-VL-7B, Qwen3-VL-8B, Gemma-3-12B, Gemma4-12B, InternVL3.5-8B (A 파일에서 2개 모델 18,000행 + B 파일에서 3개 모델 27,000행 = 총 45,000행)
2. **`margin-diagnostic set`**: Qwen2.5-VL-7B, Gemma-3-12B, InternVL3-8B, Ministral-3-8B, Phi-4-multimodal (A 파일 전체 = 총 45,000행)

---

## 2. 검증 결과

두 집합에 대해 content margin $c$와 position margin $b$의 절대값이 일치하는 경계 사례($|c| = |b|$) 및 그 중 순서 반전 일치(동일 스니펫 선택, 즉 `valid == True`) 건수를 전수 계측하였습니다.

| 집합 명칭 | 구성 파일 및 모델 | 총 관측 행 수 | $|c| = |b|$ 경계 건수 | 경계 중 실제 유효(`valid`) 건수 | 비고 |
|:---|:---|:---:|:---:|:---:|:---|
| **`margin-diagnostic set`** | `results/rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv` (A 전수)<br>(Qwen2.5, Gemma-3, InternVL3, Ministral, Phi-4) | **45,000** | **5,577건** | **2,576건** | **원고 기술 수치와 100% 완벽 일치** |
| **`primary RQ1 set`** | A 파일 2개 모델 + B 파일 3개 모델<br>(Qwen2.5, Qwen3, Gemma-3, Gemma4, InternVL3.5) | **45,000** | 1,150건 | 924건 | 불일치 |

---

## 3. 결론 및 원고 정합성 확정

1. **원고 3.5절 45,000건의 정체**: 원고 3.5절의 "45,000 reported RQ1 observations"는 **`margin-diagnostic set` (초기 5개 오픈 모델 배터리 45,000쌍)**에서 도출된 수치임이 수학적·데이터적으로 명확히 확정되었습니다.
2. **원고 서술 유지 및 명시 권고**:
   - 3.5절의 수치(5,577건 및 2,576건)는 정확한 실제 관측치이므로 수정할 필요가 없습니다.
   - 단, 독자의 혼선을 방지하기 위해 해당 문장의 대상을 "Across the 45,000 observations from the initial five-model diagnostic battery..." 또는 "margin-diagnostic set"으로 명확히 한정할 것을 논문 에이전트에게 제안합니다.
