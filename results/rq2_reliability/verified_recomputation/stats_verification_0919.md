# FSE 2027 통계 검증 보고서 (최종)

이 문서는 FSE 2027 투고 원고 "How Reliable Are VLM Judges of Rendered Code Readability?" 의 4.5절 통계 검증 및 수치 일치 여부를 종합한 최종 결과입니다. 명시적인 스크립트나 결과 파일로 확인되지 않은 모든 항목은 "미확인"으로 분류되었습니다.

## 1. 신뢰구간 및 유의성 검정 요약 (수치별 분리)

| 통계 / 수치 | 출처 파일 (결과물) | 생성 스크립트 | 구간 방법 | Cluster 단위 | Endpoint 처리 / 반복 횟수 | 상태 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RQ1 (Qwen vs RF / Best Source)** | `E1_debiased_vs_baseline.csv` | `analyze_e1.py` | Two-way snippet-cluster bootstrap | Snippet | 양쪽 인덱스 곱 가중치 (`mult[:, left] * mult[:, right]`), 10,000회 | 확인됨 |
| **RQ2 Rendering/Order (ΔE, ΔS)** | `*_matched_contrasts.csv` | `analyze_rq2_reduced.py` | Cluster-Robust SE (CRSE) | Snippet | `meat(g1)+meat(g2)-meat(g1\|\|g2)`, 샌드위치 추정량 | 확인됨 |
| **RQ4 OCR vs VLM (ΔE)** | (표준 출력) | `scratch_compute_ci.py` | Asymptotic SE | 없음 (Pair) | 쌍 단위 분산 차감 공식 ($SE$) | 확인됨 (일부)* |
| **Closed-Model Pilot (300 Pairs)** | (표준 출력) | `task9_figure5_closed_bootstrap.py` | Snippet-cluster Bootstrap | Snippet | 양쪽 인덱스 곱 가중치 (`mult[:, left] * mult[:, right]`), 10,000회 | 확인됨 |

*주석: `scratch_compute_ci.py`는 Java 데이터가 없어 에러가 발생하며, Asymptotic SE를 사용합니다.

---

## 2. 수치별 상세 검증 및 재계산 (Re-calculated 0919)

### 2.1 Full-Grid 6 Group 의 Order vs Rendering (최우선 재계산)
*   **Unique Conditions:** 원본 파일 검사 결과 총 12개의 조건이 존재함을 확인.
*   **Median Flips (Valid-both):** 66개 조건 쌍 $\binom{12}{2}$ 위에서 `analyze_rq2_reduced.py` 와 동일한 교집합 규칙(pair_id) 및 valid 마스킹을 적용해 계산.
*   **결과 반영:** 계산된 수치(Median Flip Rate 및 `order_exceeds_rendering` boolean)를 `rq2_order_vs_rendering.csv` 의 "uncalculated" 위치에 정상적으로 채웠습니다.

### 2.2 RQ1 Qwen2.5-VL-7B (Anchor) Invalidity 재분해
RQ1 실행 9,000개 쌍(총 18,000번 호출)의 `raw.jsonl` 데이터를 분석한 결과:
*   **Parsing Failure:** 두 호출 모두 존재하며 `parsed_choice` 값이 모두 A/B로 파싱 성공한 것으로 **확인됨**. (추가로 RQ2 grid raw 72,000건에 대한 `grep` 탐색 결과에서도 결측치가 0건임이 확인됨).
*   **Swap Error (45.78%):** 두 순서 모두 파싱 성공 후 선택이 뒤바뀐 "Parsed disagreement" 현상으로 **확인됨**.

### 2.3 미보고 및 원고 불일치 수치 정리 (2026-09-20 갱신)

> **갱신 공지 (2026-09-20)**: 아래 1, 2번 항목은 당시 **미확인**으로 남겼으나, 이후 근거를 모두 찾아
> **해소되었습니다**. 상세는 `notes/verify_0920_D_intervals.md` 및 `results/verified/recomputed_intervals.csv` 참조.
> 당시 탐색이 실패한 원인은 범위 오류입니다: 현재 원고의 Figure 7 은 2026-09-12 번호 체계의 **Figure 4**라
> 파일명이 `fig4_data.csv` 이고, 산출물 번들이 `experiment_26_v1/` **바깥**인
> `/ANON/home/fse2027_handoff_rq1_20260912/` 에 있어
> `grep -rn "figure7" /ANON/experiment_root/` 류의 탐색에 걸리지 않았습니다.

1. ~~**RQ4 (Java/Python) 구간:** 스크립트를 발견하지 못함.~~
   → **해소됨.** `fse2027/external_runs/deploy_v2_logit_20260731/analysis/F5_deploy_debiased.csv` 의
   `ci_lo`/`ci_hi` 열이 Java `[-2.19, +10.65]`, Python `[-5.91, +12.76]` 과 완전 일치.
   방법은 `F5_ANALYSIS_AUDIT.json`: "10,000 two-endpoint snippet-cluster replicates; dyad weight=w_i*w_j".
   `F5_pair_level_metrics.csv` 의 `d_correct - ocr_correct` 로 점추정치(+4.20 / +3.53pp)를 **완전 재현**했고,
   구간은 원 RNG 스트림 순서를 복원하지 못해 ±0.1pp 몬테카를로 오차 내에서 일치합니다.

2. ~~**Fig 7 D 구간:** D 값의 신뢰 구간을 산출하는 별도 스크립트 발견 못 함.~~
   → **해소됨.** 생성 스크립트는 `generate_rq1_bundle.py` (repo root) L140-235
   "Part 2b: 10,000-Resample Snippet-Cluster Bootstrap" 이며, D 는 `valid`/`E`/`S` 와
   **동일한 루프·동일한 가중치**(`w = mult[:, left] * mult[:, right]`, seed 42, 552 clusters)로 계산됩니다.
   산출물은 `/ANON/home/fse2027_handoff_rq1_20260912/rq1_intervals.csv` 및 `fig_data/fig4_data.csv`.
   원자료에서 8개 모델 pooled D 를 재계산해 **전부 atol=1e-12 수준으로 일치**함을 확인했습니다.
   당시 "pair 단위 D 값 파일을 찾을 수 없음" 이라는 판단은 **오류**입니다 —
   상류 `pair_level_results.csv`(45,000행 = 5 judge x 9,000)와
   `primary_pair_level.csv`(27,000행 = 3 judge x 9,000)의 **`debiased_correct` 열**이 그것입니다.

3. **18 Conditions / 108,000건:** Grid baseline과 Perturbation baseline은 파일과 결과(Verdict 일치율 100%)가 완전히 동일한 중복 파일임. 따라서 실제 고유 조건은 17개. *(D 무관 항목 — 갱신 없음, 그대로 유효)*

4. **[신규 2026-09-20] RQ4 CUDA 구간:** Table 3b 의 CUDA 비교군은 **EasyOCR + linear regression (61.37)** 인데
   `F5_deploy_debiased.csv` 는 **RapidOCR + LR (60.73)** 을 사용합니다.
   `results/python_cuda_missing_experiments_20260716/ocr/ocr_classical_pair_predictions.csv` 의
   pair 단위 `is_correct` 를 결합해(새 추론 불필요, pair_id 3,000개 완전 일치)
   Table 3b 비교군 기준 구간을 새로 계산했습니다: **+4.20pp, 95% CI [-2.6, +11.0]**.
