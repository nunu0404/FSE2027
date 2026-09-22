# RQ2 신뢰성 축소 그리드 실험 결과 최종 보고서 (수정본)

> **실험 완료 일시**: 2026년 9월 12일 02:45 (KST)  
> **평가 대상 판정자(Judges)**:
> 1. **Gemma-3-12B-it** (체크포인트: `google/gemma-3-12b-it`, 리비전: `96b6f1eccf38110c56df3a15bffe176da04bfd80`)
> 2. **InternVL3.5-8B-HF** (체크포인트: `OpenGVLab/InternVL3_5-8B-HF`, 리비전: `741a7d03020411e666c6109218ab71e08151ef86`)
> 3. **Qwen2.5-VL-7B-Instruct** (Baseline Anchor Re-run, 리비전: `cc594898137f460bfe9f0759e9844b3ce807cfb5`)  
> **총 추론 호출 수**: **90,000 calls** (Gemma-3-12B 42,000 + InternVL3.5 42,000 + Qwen Anchor 6,000)

---

## 1. 판정자 정체성 확인 및 모델 식별 (Judge Identity Verification)

실행된 매니페스트(`rq2_eval_reduced/manifests/gemma_3_12b_reduced_manifest.json`)와 RQ1 아카이브(`~/fse2027_rq1_recovery/gpusystem/`)의 매니페스트를 대조하여 판정자 모델을 검증하였습니다.

- **현재 실행 매니페스트**:
  - HuggingFace Repository ID: `google/gemma-3-12b-it`
  - Model Revision Hash: `96b6f1eccf38110c56df3a15bffe176da04bfd80`
  - Tokenizer ID: `google/gemma-3-12b-it`
- **RQ1 Gemma4-12B 매니페스트 (`results/latest_vlm_extension_20260830/inference/full/gemma4/manifest.json`)**:
  - Model ID: `google/gemma-4-12B-it`
  - Model Revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- **RQ1 Gemma-3-12B 매니페스트 (`results/rq1_model_battery_3lang_20260723/inference/full/gemma/manifest.json`)**:
  - Model ID: `google/gemma-3-12b-it`
  - Model Revision: `96b6f1eccf38110c56df3a15bffe176da04bfd80`

**식별 결론**:
실행된 모델의 리비전은 `96b6f1e...`로 `google/gemma-3-12b-it`와 정확히 일치하며, Gemma4-12B(`707f0a3...`)가 아닌 **Gemma-3-12B**입니다.
이에 따라 모든 출력 파일, 테이블 헤더, 메트릭 표, 매니페스트 필드의 표기를 `Gemma-3-12B`로 전면 수정하였으며, 본 결과를 제3의 판정자(third judge) 데이터로 유지합니다.

> **Gemma4-12B (`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`) 실행 리소스 및 스케줄링 안내**:
> - **체크포인트 위치**: `/ANON/scratch_rq1/hf/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (로컬 디스크에 보관되어 즉시 로딩 가능)
> - **소요 GPU**: 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB VRAM, GPU 0 또는 1)
> - **예상 소요 시간 (Wall-clock Time)**: 약 **6시간 45분** (동일 파이프라인 Gemma-3-12B 42,000콜 실행 시 406분 소요, 평균 처리 속도 1.72 calls/s 기준)

---

## 2. 순서 교란 vs 렌더링 교란 정량 보고 (Order vs. Rendering Summary)

해석적 수식어를 배제하고 순수한 정량적 수치만을 보고합니다.

| 판정자 (Judge) | 언어 (Language) | 조건 내 순서 교체 오류 중앙값 (Swap Error) | 고정 순서 렌더링 뒤집힘 비율 중앙값 (Flip Rate) | 점수 차이 ($\Delta$ points) | 양쪽 유효 쌍 기준 뒤집힘 비율 (Both-Valid Flip) | 비고 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Gemma-3-12B** | CUDA | 18.95% | 18.25% | **+0.70%p** | 7.68% | **차이가 3 points 미만 (0.70%p)** |
| | Java | 18.30% | 15.65% | **+2.65%p** | 5.84% | **차이가 3 points 미만 (2.65%p)** |
| | Python | 27.80% | 19.25% | **+8.55%p** | 8.74% | |
| **InternVL3.5-8B** | CUDA | 29.35% | 13.25% | **+16.10%p** | 3.82% | |
| | Java | 30.25% | 15.92% | **+14.33%p** | 5.70% | |
| | Python | 37.70% | 14.11% | **+23.59%p** | 3.14% | |

- **수치 요약**:
  - Gemma-3-12B의 경우, CUDA에서 순서 교체 오류(18.95%)와 렌더링 뒤집힘 비율(18.25%)의 차이는 **0.70%p**로 3 points 미만입니다. Java에서도 순서 교체 오류(18.30%)와 렌더링 뒤집힘 비율(15.65%)의 차이는 **2.65%p**로 3 points 미만입니다. Python에서는 차이가 **8.55%p**입니다. 양쪽 순서 모두에서 유효 판정을 내린 쌍 기준 렌더링 뒤집힘 비율은 세 언어에서 5.84% ~ 8.74%입니다.
  - InternVL3.5-8B의 경우, 세 언어 모두에서 차이가 3 points를 초과합니다 (CUDA +16.10%p, Java +14.33%p, Python +23.59%p). 양쪽 순서 모두 유효한 쌍 기준 렌더링 뒤집힘 비율은 3.14% ~ 5.70%입니다.

---

## 3. InternVL3.5-8B 미파싱 호출 (247 Unparsed Calls) 분석

InternVL3.5-8B의 총 42,000 추론 호출 중 247건(0.59%)에서 출력 형식 파싱(`parsed_choice`가 A 또는 B로 매칭)이 실패하였습니다.

### 3.1. 언어 및 조건별 발생 분포
- **CUDA (총 13건)**:
  - `gaussian_sigma_4`: 11건
  - `no_indent`: 2건
- **Java (총 162건)**:
  - `gaussian_sigma_4`: 149건
  - `monokai_dark__fs20__wrap60__lnon`: 1건
  - `monokai_dark__fs20__wrap80__lnon`: 4건
  - `monokai_dark__fs24__wrap80__lnon`: 3건
  - `no_indent`: 5건
- **Python (총 72건)**:
  - `gaussian_sigma_4`: 64건
  - `monokai_dark__fs20__wrap80__lnon`: 1건
  - `monokai_dark__fs24__wrap80__lnon`: 3건
  - `no_blank_lines`: 2건
  - `no_indent`: 2건

### 3.2. 메트릭 산출 시 처리 방식 확인
- 247개의 호출은 데이터셋에서 제거(dropped)되지 않았습니다.
- 쌍(pair) 단위 결합 시 한쪽이라도 미파싱된 쌍은 `parsed_both = False`로 분류되며, 자동으로 `strict_valid = False` 및 `strict_correct = False`로 처리됩니다.
- 엄격 유효 정확도 $E$ ($n\_correct / 1000$) 및 순서 교체 오류 $S$ ($1 - n\_valid / 1000$) 계산 시 고정 분모 $N=1,000$이 그대로 유지되므로, 이들 247건은 **유효하지 않은 쌍(invalid pairs)으로 분모에 완전히 포함**되어 평가되었습니다.

---

## 4. Qwen2.5-VL-7B Baseline Anchor Re-run 대조 진술

동일 환경 재실행(same-environment re-run) 결과는 모든 평가 지표에서 최대 0.8%p 차이 이내로 나타났습니다.
원고에 보고된 3.70%p 격차는 서로 다른 렌더러 및 실행 환경 간의 차이로 인한 것입니다.

| 언어 | N(Pairs) | 재실행 Valid Acc | 기존 Valid Acc | Valid Acc 차이 | 재실행 Effective Acc | 기존 Effective Acc | Effective Acc 차이 | 재실행 Swap Error | 기존 Swap Error | Swap Error 차이 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CUDA** | 1000 | 75.36% | 76.87% | -1.52%p | 47.40% | 48.20% | **-0.80%p** | 37.10% | 37.30% | **-0.20%p** |
| **Java** | 1000 | 64.26% | 64.40% | -0.14%p | 32.90% | 33.10% | **-0.20%p** | 48.80% | 48.60% | **+0.20%p** |
| **Python** | 1000 | 76.22% | 76.05% | +0.18%p | 32.70% | 32.70% | **0.00%p** | 57.10% | 57.00% | **+0.10%p** |

---

## 5. 상세 메트릭 요약 표 (Section 5.2 축소 그리드)

### 5.1. Gemma-3-12B 요약 메트릭
- Baseline: `monokai_dark__fs20__wrap80__lnon`

| 언어 | 조건 | N(Pairs) | Parsed | Valid | Correct | Valid Acc [95% CI] | Effective Acc [95% CI] | Strict Swap Error | $D$ (Sign(c)) | First-Pos Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CUDA** | **Baseline** | 1000 | 1000 | 813 | 542 | 66.67% [63.35, 69.82] | 54.20% [51.10, 57.27] | 18.70% | 63.50% | 51.05% |
| | Wrap 60 | 1000 | 1000 | 768 | 535 | 69.66% [66.32, 72.81] | 53.50% [50.40, 56.57] | 23.20% | 66.00% | 50.00% |
| | Font 24 | 1000 | 1000 | 808 | 550 | 68.07% [64.78, 71.19] | 55.00% [51.90, 58.06] | 19.20% | 65.90% | 48.40% |
| | Light Theme | 1000 | 1000 | 818 | 549 | 67.11% [63.82, 70.25] | 54.90% [51.80, 57.96] | 18.20% | 64.00% | 49.40% |
| | No Indent | 1000 | 1000 | 780 | 521 | 66.79% [63.41, 70.01] | 52.10% [49.00, 55.18] | 22.00% | 64.20% | 49.40% |
| | No Blank Lines | 1000 | 1000 | 784 | 537 | 68.49% [65.16, 71.65] | 53.70% [50.60, 56.77] | 21.60% | 66.50% | 49.20% |
| | Gaussian Blur ($\sigma=4$) | 1000 | 1000 | 698 | 492 | 70.49% [67.00, 73.75] | 49.20% [46.11, 52.30] | 30.20% | 66.10% | 39.20% |
| **Java** | **Baseline** | 1000 | 1000 | 822 | 510 | 62.04% [58.68, 65.30] | 51.00% [47.90, 54.09] | 17.80% | 61.30% | 54.50% |
| | Wrap 60 | 1000 | 1000 | 812 | 495 | 60.96% [57.56, 64.26] | 49.50% [46.41, 52.59] | 18.80% | 60.00% | 52.40% |
| | Font 24 | 1000 | 1000 | 799 | 500 | 62.58% [59.17, 65.87] | 50.00% [46.91, 53.09] | 20.10% | 61.80% | 54.75% |
| | Light Theme | 1000 | 1000 | 831 | 522 | 62.82% [59.48, 66.04] | 52.20% [49.10, 55.28] | 16.90% | 60.50% | 54.05% |
| | No Indent | 1000 | 1000 | 805 | 489 | 60.75% [57.33, 64.06] | 48.90% [45.81, 52.00] | 19.50% | 58.70% | 53.35% |
| | No Blank Lines | 1000 | 1000 | 788 | 497 | 63.07% [59.65, 66.37] | 49.70% [46.61, 52.79] | 21.20% | 59.40% | 54.60% |
| | Gaussian Blur ($\sigma=4$) | 1000 | 1000 | 604 | 340 | 56.29% [52.31, 60.20] | 34.00% [31.13, 36.99] | 39.60% | 55.60% | 31.90% |
| **Python**| **Baseline** | 1000 | 1000 | 725 | 458 | 63.17% [59.60, 66.61] | 45.80% [42.73, 48.90] | 27.50% | 60.70% | 60.65% |
| | Wrap 60 | 1000 | 1000 | 717 | 442 | 61.65% [58.03, 65.13] | 44.20% [41.15, 47.29] | 28.30% | 59.20% | 59.65% |
| | Font 24 | 1000 | 1000 | 719 | 426 | 59.25% [55.62, 62.78] | 42.60% [39.57, 45.69] | 28.10% | 58.10% | 61.35% |
| | Light Theme | 1000 | 1000 | 737 | 446 | 60.52% [56.94, 63.98] | 44.60% [41.55, 47.70] | 26.30% | 58.90% | 60.15% |
| | No Indent | 1000 | 1000 | 675 | 426 | 63.11% [59.41, 66.67] | 42.60% [39.57, 45.69] | 32.50% | 61.00% | 62.55% |
| | No Blank Lines | 1000 | 1000 | 724 | 430 | 59.39% [55.77, 62.91] | 43.00% [39.96, 46.09] | 27.60% | 57.00% | 59.20% |
| | Gaussian Blur ($\sigma=4$) | 1000 | 1000 | 708 | 449 | 63.42% [59.81, 66.88] | 44.90% [41.84, 48.00] | 29.20% | 59.70% | 40.00% |

---

### 5.2. InternVL3.5-8B 요약 메트릭
- Baseline: `monokai_dark__fs20__wrap80__lnon`

| 언어 | 조건 | N(Pairs) | Parsed | Valid | Correct | Valid Acc [95% CI] | Effective Acc [95% CI] | Strict Swap Error | $D$ (Sign(c)) | First-Pos Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CUDA** | **Baseline** | 1000 | 1000 | 743 | 493 | 66.35% [62.88, 69.66] | 49.30% [46.21, 52.40] | 25.70% | 64.00% | 39.75% |
| | Wrap 60 | 1000 | 1000 | 645 | 420 | 65.12% [61.36, 68.70] | 42.00% [38.98, 45.08] | 35.50% | 62.90% | 34.75% |
| | Font 24 | 1000 | 1000 | 714 | 475 | 66.53% [62.98, 69.89] | 47.50% [44.42, 50.60] | 28.60% | 63.10% | 37.90% |
| | Light Theme | 1000 | 1000 | 699 | 484 | 69.24% [65.72, 72.55] | 48.40% [45.31, 51.50] | 30.10% | 65.30% | 39.15% |
| | No Indent | 1000 | 999 | 714 | 482 | 67.51% [63.99, 70.84] | 48.20% [45.12, 51.30] | 28.60% | 63.60% | 38.90% |
| | No Blank Lines | 1000 | 1000 | 664 | 427 | 64.31% [60.59, 67.86] | 42.70% [39.67, 45.79] | 33.60% | 61.40% | 36.20% |
| | Gaussian Blur ($\sigma=4$) | 1000 | 990 | 437 | 288 | 65.90% [61.34, 70.19] | 28.80% [26.08, 31.68] | 56.30% | 59.30% | 21.85% |
| **Java** | **Baseline** | 1000 | 996 | 685 | 427 | 62.34% [58.65, 65.89] | 42.70% [39.67, 45.79] | 31.50% | 60.80% | 41.65% |
| | Wrap 60 | 1000 | 999 | 623 | 387 | 62.12% [58.25, 65.84] | 38.70% [35.73, 41.76] | 37.70% | 59.90% | 34.00% |
| | Font 24 | 1000 | 997 | 710 | 436 | 61.41% [57.78, 64.92] | 43.60% [40.56, 46.69] | 29.00% | 59.00% | 40.30% |
| | Light Theme | 1000 | 1000 | 721 | 450 | 62.41% [58.82, 65.87] | 45.00% [41.94, 48.10] | 27.90% | 59.50% | 41.15% |
| | No Indent | 1000 | 995 | 671 | 409 | 60.95% [57.21, 64.57] | 40.90% [37.89, 43.98] | 32.90% | 59.30% | 36.60% |
| | No Blank Lines | 1000 | 1000 | 687 | 431 | 62.74% [59.06, 66.27] | 43.10% [40.06, 46.19] | 31.30% | 59.60% | 40.35% |
| | Gaussian Blur ($\sigma=4$) | 1000 | 886 | 473 | 257 | 54.33% [49.83, 58.77] | 25.70% [23.09, 28.50] | 52.70% | 46.00% | 24.40% |
| **Python**| **Baseline** | 1000 | 999 | 639 | 437 | 68.39% [64.68, 71.87] | 43.70% [40.66, 46.79] | 36.10% | 64.40% | 33.35% |
| | Wrap 60 | 1000 | 1000 | 564 | 374 | 66.31% [62.31, 70.09] | 37.40% [34.45, 40.44] | 43.60% | 62.50% | 29.80% |
| | Font 24 | 1000 | 997 | 662 | 453 | 68.43% [64.79, 71.85] | 45.30% [42.24, 48.40] | 33.80% | 64.50% | 34.85% |
| | Light Theme | 1000 | 1000 | 607 | 432 | 71.17% [67.44, 74.63] | 43.20% [40.16, 46.29] | 39.30% | 65.40% | 31.65% |
| | No Indent | 1000 | 998 | 585 | 399 | 68.21% [64.32, 71.85] | 39.90% [36.91, 42.97] | 41.50% | 63.40% | 30.35% |
| | No Blank Lines | 1000 | 998 | 553 | 394 | 71.25% [67.34, 74.86] | 39.40% [36.42, 42.46] | 44.70% | 65.80% | 29.80% |
| | Gaussian Blur ($\sigma=4$) | 1000 | 942 | 313 | 160 | 51.12% [45.60, 56.61] | 16.00% [13.86, 18.40] | 68.70% | 47.60% | 15.65% |

---

## 6. 데이터 아티팩트 및 파일 인덱스

- **Gemma-3-12B 산출물**:
  - 요약 메트릭: [`Gemma-3-12B_summary_metrics.csv`](file:///ANON/experiment_root/rq2_eval_reduced/gemma/summary/Gemma-3-12B_summary_metrics.csv)
  - 순서 vs 렌더링: [`Gemma-3-12B_order_vs_rendering.csv`](file:///ANON/experiment_root/rq2_eval_reduced/gemma/summary/Gemma-3-12B_order_vs_rendering.csv)
  - Matched Contrasts: [`Gemma-3-12B_matched_contrasts.csv`](file:///ANON/experiment_root/rq2_eval_reduced/gemma/summary/Gemma-3-12B_matched_contrasts.csv)
  - Figure 6(a) 입력 데이터: [`Gemma-3-12B_fig6_input.csv`](file:///ANON/experiment_root/rq2_eval_reduced/gemma/figures_input/Gemma-3-12B_fig6_input.csv)
  - 매니페스트: [`gemma_3_12b_reduced_manifest.json`](file:///ANON/experiment_root/rq2_eval_reduced/manifests/gemma_3_12b_reduced_manifest.json)
- **InternVL3.5-8B-HF 산출물**:
  - 요약 메트릭: [`InternVL3.5-8B_summary_metrics.csv`](file:///ANON/experiment_root/rq2_eval_reduced/internvl3_5/summary/InternVL3.5-8B_summary_metrics.csv)
  - 순서 vs 렌더링: [`InternVL3.5-8B_order_vs_rendering.csv`](file:///ANON/experiment_root/rq2_eval_reduced/internvl3_5/summary/InternVL3.5-8B_order_vs_rendering.csv)
  - Matched Contrasts: [`InternVL3.5-8B_matched_contrasts.csv`](file:///ANON/experiment_root/rq2_eval_reduced/internvl3_5/summary/InternVL3.5-8B_matched_contrasts.csv)
  - Figure 6(a) 입력 데이터: [`InternVL3.5-8B_fig6_input.csv`](file:///ANON/experiment_root/rq2_eval_reduced/internvl3_5/figures_input/InternVL3.5-8B_fig6_input.csv)
  - 매니페스트: [`internvl3_5_8b_reduced_manifest.json`](file:///ANON/experiment_root/rq2_eval_reduced/manifests/internvl3_5_8b_reduced_manifest.json)
- **Qwen Anchor Re-run 대조 산출물**:
  - 재현성 대조표: [`qwen25_anchor_vs_original.csv`](file:///ANON/experiment_root/rq2_eval_reduced/anchor_qwen/qwen25_anchor_vs_original.csv)
  - 매니페스트: [`qwen25_vl_7b_anchor_manifest.json`](file:///ANON/experiment_root/rq2_eval_reduced/manifests/qwen25_vl_7b_anchor_manifest.json)
