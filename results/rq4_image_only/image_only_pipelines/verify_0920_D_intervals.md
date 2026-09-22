# B1 검증 결과: Figure 7 의 D whisker 계산 근거 (확인됨)

**검증일**: 2026-09-20 / **결론**: D whisker 는 실제 계산된 값임. **Figure 7 유지, 4.5 본문 수정** 으로 처리.

## 1. 계산 근거 (모두 확인됨)

| 항목 | 경로 |
| :--- | :--- |
| 생성 스크립트 | `generate_rq1_bundle.py` (repo root), L140–235 "Part 2b: 10,000-Resample Snippet-Cluster Bootstrap" |
| 그림용 데이터 | `/ANON/home/fse2027_handoff_rq1_20260912/fig_data/fig4_data.csv` (열: `D`, `D_lo`, `D_hi`) |
| 전체 구간표 | `/ANON/home/fse2027_handoff_rq1_20260912/rq1_intervals.csv` (`metric==D`, language=java/python/cuda/pooled) |
| 출처/체크섬 | `/ANON/home/fse2027_handoff_rq1_20260912/SOURCES.md` — 입력 3개 + 산출물 SHA-256 **전부 일치 재확인** |

**중요**: 현재 원고 Figure 7 = 2026-09-12 번호 체계의 **Figure 4**. 파일명이 `fig4_data.csv` 이고
번들이 `experiment_26_v1/` **밖**(`/ANON/home/`)에 있어 이전 라운드의
`grep -rn "figure7" /ANON/experiment_root/` 탐색에서 누락되었음.

## 2. 계산 방법 (코드 실제 내용)

```python
mult = rng.multinomial(len(snippets), probs, size=10000)   # seed 42, 552 snippet clusters
w    = mult[:, left] * mult[:, right]                      # 두 endpoint 가중치의 곱
d_point = deb_arr.sum() / n_p                              # 전체 9,000 쌍 분모
d_reps  = (w * deb_arr).sum(axis=1) / w_tot
"D": (d_point, np.quantile(d_reps, 0.025), np.quantile(d_reps, 0.975))
```

- D 는 `valid` / `E` / `S` 와 **완전히 동일한 루프·동일한 가중치**로 계산됨.
- `deb_arr` = 상류 pair-level 파일의 `debiased_correct` (bool). E1 파일의 대응 열
  `debiased_main_tie_incorrect` 로 보아 **tie = 오답** 규약(3.3/3.4절 D 정의)과 일치.
- 4.5절의 "product of its two snippet weights, jointly resampling both endpoints" 서술과 코드가 정확히 일치.

## 3. 독립 재현 (원자료에서 재계산)

8개 모델 pooled D 점추정치·상한·하한 **전부 `rq1_intervals.csv` 와 atol=1e-12 수준 완전 일치**.

| judge | D [95% CI] |
| :--- | :--- |
| Gemma4-12B | 64.62 [61.34, 67.78] |
| Qwen2.5-VL-7B | 64.16 [60.79, 67.37] |
| InternVL3.5-8B | 62.30 [59.07, 65.47] |
| Gemma-3-12B | 62.01 [58.63, 65.13] |
| Qwen3-VL-8B | 59.91 [56.67, 63.17] |
| Ministral-3-8B | 54.30 [50.90, 57.72] |
| InternVL3-8B | 51.00 [47.52, 54.50] |
| Phi-4-multimodal | 37.83 [34.80, 40.97] |

재현 스크립트: `scratchpad/verify_D_ci.py`. 입력 원자료(체크섬 확인됨):
- `.../rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv` (9,000 쌍)
- `.../rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv` (45,000행 = 5 judge x 9,000)
- `.../latest_vlm_extension_20260830/analysis/primary_pair_level.csv` (27,000행 = 3 judge x 9,000)

→ 0919 노트의 "Fig 7 의 8개 모델에 대한 pair 단위 D 값 파일을 찾을 수 없음" 은 **사실과 다름**. 위 두 파일의 `debiased_correct` 열이 그것임.

## 4. 본문이 이미 D 구간을 보고하고 있음

5.1.1절: "Both clustered intervals ([-4.31, +12.04] and [-6.18, +10.23]) include zero."
→ 출처 `fse2027/external_runs/fse2027_review_defense_e1_e8_20260730/analysis/E1/E1_debiased_vs_baseline.csv`,
  Qwen2.5 / python 행: `diff_vs_rf_ci_low/high` = **[-4.31, +12.04]**,
  `diff_vs_language_best_ci_low/high` = **[-6.18, +10.23]** (완전 일치).
  같은 행 `bootstrap_reps=10000`, `bootstrap_cluster="two-way snippet multiplicity"`.
즉 D 는 그림뿐 아니라 **본문에서도 이미 구간과 함께 보고**되고 있음.

## 5. 4.5절 수정안 (Figure 7 은 그대로 유지)

**(a) 두 번째 문장 — D 를 목록에 추가**
> RQ1 uses 10,000 snippet-cluster bootstrap replicates for valid accuracy, effective accuracy,
> **two-order averaged accuracy**, strict-swap error, and differences against source-feature baselines.
> Intervals are 2.5th/97.5th percentiles of the replicate distribution (seed 42).

**(b) 마지막 문장 — RQ4 로 한정**
> ~~We report RQ4 pipeline differences and two-order averaged accuracy as point estimates without intervals.~~
> → We report RQ4 pipeline differences and **the RQ4 deployment** two-order averaged accuracy
>   as point estimates without intervals.

Figure 7 캡션("Whiskers are reported 95% snippet-cluster bootstrap intervals")은 **수정 불필요** — E/D/S 3개 지표 모두 `fig4_data.csv` 에 구간이 있고 검증됨.

## 6. 부수 발견 (별건, 참고용)

1. **RQ4 구간도 실재함.** `deploy_v2_logit_20260731/analysis/F5_deploy_debiased.csv` 의 `ci_lo/ci_hi`:
   Java **[-2.19, +10.65]**, Python **[-5.91, +12.76]**, CUDA [-2.64, +12.22].
   `F5_ANALYSIS_AUDIT.json`: "10,000 two-endpoint snippet-cluster replicates; dyad weight=w_i*w_j".
   단, 이 구간의 비교 대상은 `RapidOCR + best ML` 이며 **Java/Python 은 Table 3b 와 동일 비교군**
   (52.97 / 59.53), **CUDA 만 다름** (파일 60.73 vs Table 3b 61.37 EasyOCR+LR).
   → RQ4 를 점추정치로만 보고하는 현재 선택은 유지 가능하나, Java/Python 은 원하면 구간 제시 가능.
2. **Table 3b Python 차이값 반올림.** 63.0667-59.5333 = **3.53**pp 인데 원고는 +3.54 (반올림된 값끼리 뺀 결과). 0.01pp 불일치.

## 7. 선행 문서 중 갱신 필요 항목

- `notes/stats_verification_0919.md` 2.3절 항목 1, 2 → 본 문서로 **해소됨**
- `results/verified/recomputed_intervals.csv` 4개 행 전부 "미계산/미확인" → 본 문서로 **해소됨**
(두 파일은 임의 수정하지 않았음. 갱신 여부는 지시 필요.)

---

# 갱신 로그 (2026-09-20, 갈래 A 확정 후속)

## RQ4 CUDA paired difference 구간 — 신규 계산 (새 추론 불필요)

Table 3b 의 CUDA 비교군은 `EasyOCR + linear regression` (61.37) 인데,
`F5_deploy_debiased.csv` 는 `RapidOCR + linear regression` (60.73) 을 썼기 때문에 구간이 없었음.
기존 pair-level 결과만으로 계산 가능함을 확인하고 계산함.

- 결합: `F5_pair_level_metrics.csv`(`d_correct`) x
  `results/python_cuda_missing_experiments_20260716/ocr/ocr_classical_pair_predictions.csv`(`is_correct`)
- 검증: pair_id 3,000개 집합 완전 일치 / `F5.ocr_correct == rapidocr+LR.is_correct` 전 행 일치로 비교군 정체 확인
- 방법: 두-endpoint snippet-cluster paired bootstrap, dyad weight `w_i*w_j`, 언어별 snippet pool
- 재현 검증: 기존 공표 구간(Java/Python) 점추정치 **완전 일치**, 구간은 ±0.1pp 내 일치
  (원 RNG 스트림 순서 미복원 — 100k/200k reps, seed 42/7/2027 간 변동 ±0.06pp)

| language | Table 3b 비교군 | OCR+ML | D | diff | 95% CI |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Java | RapidOCR + SVR | 52.97 | 57.17 | +4.20 | [-2.3, +10.6] (공표 [-2.19, +10.65]) |
| Python | RapidOCR + gradient boosting | 59.53 | 63.07 | +3.53 | [-5.9, +12.9] (공표 [-5.91, +12.76]) |
| CUDA | **EasyOCR + linear regression** | 61.37 | 65.57 | +4.20 | **[-2.6, +11.0] (신규)** |

**세 언어 모두 0 을 포함** → 5.4.1 절의 "establish neither superiority nor equivalence" 결론은 그대로 유지됨.
(구간을 싣기로 하면 4.5 절 마지막 문장의 "RQ4 pipeline differences" 부분도 함께 조정 필요.)

## 파일 갱신 내역

### 1) `results/verified/recomputed_intervals.csv` (4행 전부 교체 + 스키마 수정)

기존 파일은 `[-2.19, 10.65]` 가 따옴표 없이 쉼표를 포함해 **CSV 파싱이 깨져 있었음**(1-2행이 5필드로 분리).
`Status` 열을 추가하고 전 필드를 정상 인용 처리함.

```
- RQ4 Java(RapidOCR+SVR) vs Qwen,[-2.19, 10.65],미계산,"inventory.csv에 RQ4 관련 기록이 없고, ... 계산 불가"
+ "RQ4 Java(RapidOCR+SVR) vs Qwen","[-2.19, +10.65]","[-2.3, +10.6]","확인됨(재현)","F5_pair_level_metrics.csv 의 d_correct - ocr_correct 로 계산. 점추정치 +4.20pp 완전 일치 ..."

- RQ4 Python(RapidOCR+GBM) vs Qwen,[-5.91, 12.76],미계산,"OCR 모델의 pair 단위 예측 결과 없음"
+ "RQ4 Python(RapidOCR+GBM) vs Qwen","[-5.91, +12.76]","[-5.9, +12.9]","확인됨(재현)","동일 방법. 점추정치 +3.53pp 완전 일치 ..."

- RQ4 CUDA(EasyOCR+LR) vs Qwen,미확인,미계산,"OCR 모델의 pair 단위 예측 결과 없음"
+ "RQ4 CUDA(EasyOCR+LR) vs Qwen","원고 미보고(점추정치 +4.20pp 만)","[-2.6, +11.0]","신규 계산됨","... ocr_classical_pair_predictions.csv 결합해 계산(새 추론 불필요) ..."

- Fig 7 D (8 models),미확인,미계산,"Fig 7 의 8개 모델에 대한 pair 단위 예측(D 값)이 기록된 파일을 찾을 수 없어 계산 불가"
+ "Fig 7 D (8 models)","fig_data/fig4_data.csv 의 D_lo/D_hi","8개 모델 전부 완전 일치(atol=1e-12)","확인됨(재현)","generate_rq1_bundle.py L140-235 ... 이전 '파일 없음' 판단은 오류"
```

### 2) `notes/stats_verification_0919.md` 2.3 절

```
- ### 2.3 미보고 및 원고 불일치 수치 정리 (미확인 목록)
+ ### 2.3 미보고 및 원고 불일치 수치 정리 (2026-09-20 갱신)

- 아래 항목들은 코드 및 디렉터리를 탐색했으나 증거를 발견하지 못하여 **미확인(Unverified)** 으로 남겨둡니다.
+ > 갱신 공지 (2026-09-20): 1, 2번 항목 해소됨. 탐색 실패 원인은 범위 오류(파일명 fig4_data.csv,
+ >  번들이 experiment_26_v1/ 바깥의 /ANON/home/fse2027_handoff_rq1_20260912/ 에 위치).

- 1. **RQ4 (Java/Python) 구간:** ... 스크립트를 발견하지 못함.
+ 1. ~~...~~ → **해소됨.** F5_deploy_debiased.csv 의 ci_lo/ci_hi 가 공표값과 완전 일치 ...

- 2. **Fig 7 D 구간:** D 값의 신뢰 구간을 산출하는 별도 스크립트 발견 못 함.
+ 2. ~~...~~ → **해소됨.** generate_rq1_bundle.py L140-235 ... debiased_correct 열이 pair 단위 D 값 ...

  3. **18 Conditions / 108,000건:** ... (D 무관 — 변경 없음)
+ 4. **[신규] RQ4 CUDA 구간:** Table 3b 비교군(EasyOCR+LR) 기준 +4.20pp, 95% CI [-2.6, +11.0]
```

백업: `scratchpad/recomputed_intervals.bak.csv`, `scratchpad/stats_verification_0919.bak.md`
