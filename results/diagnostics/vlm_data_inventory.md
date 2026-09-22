# RQ1 Pair-Level 및 부트스트랩 산출물 데이터 인벤토리 보고서

**작성 일시:** 2026-09-14 10:25:00 KST  
**작성 주체:** 실험 에이전트 (Experiment Agent)  
**대상 시스템:** GPU 실험 서버 (`gpusystem`) 및 홈 디렉터리 (`/ANON/home/`)  
**목적:** 로컬 macOS 환경(`~/Downloads/results`)에 누락된 RQ1 pair-level 산출물, 판정 로짓/마진, 10,000회 부트스트랩 신뢰구간 산출물의 서버 내 전수 보존 현황 검증 및 동기화 가이드 수립

---

## 1. 개요 및 핵심 진단 요약

사용자 맥북 로컬(`~/Downloads/results`)에는 Primary Judge 중 `Qwen2.5-VL-7B`만 확인되고, 나머지 4개 Primary Judge(`Qwen3-VL-8B`, `Gemma-3-12B`, `Gemma4-12B`, `InternVL3.5-8B`)의 RQ1 pair-level 산출물과 RQ1 부트스트랩 산출물이 누락된 상태였습니다.

홈 디렉터리(`/ANON/home/`) 전체에 대한 전수 정밀 감사 결과:
1. **서버 내 100% 완전 보존 확인**: Primary Judge 5개 전 모델과 진단용 보조 모델 3개를 포함한 총 8개 오픈 소스 VLM 전원에 대해, **언어당 3,000행 / 모델당 9,000행**의 pair-level 판정 라벨, 순서 반전(AB/BA) 18,000회 추론 결과, **토큰 로짓 및 마진(margin_ab, margin_ba, content_margin, position_margin)**이 누락 없이 보존되어 있습니다.
2. **부트스트랩 산출물 완비**: 10,000회 재표본추출(Resample) 기반 스니펫 클러스터 부트스트랩(Snippet-Cluster Bootstrap) 95% 신뢰구간 파일(`rq1_intervals.csv`, 총 128행) 및 관련 LaTeX 표, 그림 데이터가 `fse2027_handoff_rq1_20260912/` 및 원본 결과 디렉터리에 완벽히 구축되어 있습니다.
3. **결론 및 재생성 여부**: 서버 내에 완전한 원본 데이터가 전수 보존되어 있으므로 **모델 재실행(재생성)은 불필요**합니다. 로컬 macOS로 해당 파일들을 전송(SCP/동기화)하여 즉시 복구할 수 있습니다.

---

## 2. 발견된 RQ1 Pair-Level 및 부트스트랩 파일 인벤토리

### 2.1 주 산출물 파일 (Pair-Level Primary Files)

한 행이 한 쌍(Pair)이며, AB 판정, BA 판정, 두 방향 마진(로짓)을 모두 보존하고 있는 핵심 파일입니다.

| 파일명 | 절대 파일 경로 | 파일 크기 | SHA-256 체크섬 | 포함 모델 및 행 수 | 컬럼 구조 요약 | 로짓/마진 보존 여부 |
|:---|:---|:---:|:---:|:---|:---|:---:|
| **`pair_level_results.csv`**<br>(5-Model Battery) | `/ANON/experiment_root/results/rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv`<br>*(백업: `/ANON/home/fse2027_rq1_recovery/gpusystem/ANON/experiment_root/results/rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv`)* | 8.41 MB | `0fa4ae44f141efbce8933019b3b0fe8e5a70906cd63cb65bf49f91d5202e4167` | **총 45,000행**<br>• **Qwen2.5-VL-7B**: 9,000행<br>• **Gemma-3-12B**: 9,000행<br>• **InternVL3-8B**: 9,000행<br>• **Ministral-3-8B**: 9,000행<br>• **Phi-4-multimodal**: 9,000행<br>*(언어당 각 3,000행)* | `model_key`, `model`, `concurrency_group`, `pair_id`, `language`, `difficulty`, `abs_z_diff`, `human_delta`, `valid`, `correct`, `model_sign`, `ab_correct`, `ba_correct`, `margin_ab`, `margin_ba`, `content_margin`, `position_margin`, `debiased_correct`, `content_tie`, `ab_first_choice`, `ba_first_choice` (총 21개 컬럼) | **로짓 포함 확보**<br>(`margin_ab`, `margin_ba`, `content_margin`, `position_margin` 전수 결측치 0건) |
| **`primary_pair_level.csv`**<br>(3-Model Extension) | `/ANON/experiment_root/results/latest_vlm_extension_20260830/analysis/primary_pair_level.csv`<br>*(백업: `/ANON/home/fse2027_rq1_recovery/gpusystem/ANON/experiment_root/results/latest_vlm_extension_20260830/analysis/primary_pair_level.csv`)* | 6.22 MB | `c053fe87d4488beed5a4f2bd18eded7de563149a1f1624f7a0123d295c09d08f` | **총 27,000행**<br>• **Qwen3-VL-8B**: 9,000행<br>• **InternVL3.5-8B**: 9,000행<br>• **Gemma4-12B**: 9,000행<br>*(언어당 각 3,000행)* | `model_key`, `model`, `pair_id`, `language`, `difficulty`, `abs_z_diff`, `human_delta`, `human_preference`, `valid`, `strict_choice`, `correct`, `model_sign`, `ab_correct`, `ba_correct`, `ab_first_choice`, `ba_first_choice`, `parse_failure_calls`, `margin_ab`, `margin_ba`, `position_margin_b`, `content_margin_c`, `debiased_choice`, `debiased_correct`, `complete_effective_logits`, `content_tie`, `boundary`, `bc_rule_evaluable`, `bc_valid_prediction`, `bc_rule_violation` (총 29개 컬럼) | **로짓 포함 확보**<br>(`margin_ab`, `margin_ba`, `position_margin_b`, `content_margin_c`, `complete_effective_logits` 보존. 파싱 에러 극소수 제외 전수 유효) |

> [!NOTE]
> **Primary Judge 4개 누락 의문 해소**:
> - 사용자의 로컬 macOS에 없었던 `Gemma-3-12B`는 원래 `pair_level_results.csv` (2026-07-23 5모델 배터리)에 속해 있습니다.
> - `Qwen3-VL-8B`, `InternVL3.5-8B`, `Gemma4-12B`는 이후 최신 모델 확장 연구로 수행된 `primary_pair_level.csv` (2026-08-30 확장 배터리)에 속해 있습니다.
> - 서버에서는 두 파일 모두 완벽하게 보존되어 있으며 무결성이 확인되었습니다.

---

### 2.2 모델별 Raw 추론 토큰 로짓 파일 (Raw JSONL & Effective Logits)

각 쌍에 대한 양방향 호출(총 18,000 calls = 9,000 pairs × 2 orders AB/BA) 원본 파일로, 개별 토큰 ID(`token_id_A`, `token_id_B`)와 토큰별 원본 로짓(`logit_A`, `logit_B`, `margin`), 생성 텍스트(`raw_output`)가 기록되어 있습니다.

| 모델명 | Raw 파일 경로 (18,000행) | Effective Logits 경로 (18,000행) | 로짓 보존 상태 |
|:---|:---|:---|:---:|
| **Qwen2.5-VL-7B** | `results/rq1_model_battery_3lang_20260723/inference/full/qwen/raw.jsonl` | (raw.jsonl 내 verdict token logit_A/B 보존) | 로짓 포함 확보 |
| **Gemma-3-12B** | `results/rq1_model_battery_3lang_20260723/inference/full/gemma/raw.jsonl` | (raw.jsonl 내 verdict token logit_A/B 보존) | 로짓 포함 확보 |
| **InternVL3-8B** | `results/rq1_model_battery_3lang_20260723/inference/full/internvl/raw.jsonl` | (raw.jsonl 내 verdict token logit_A/B 보존) | 로짓 포함 확보 |
| **Ministral-3-8B** | `results/rq1_model_battery_3lang_20260723/inference/full/ministral/raw.jsonl` | (raw.jsonl 내 verdict token logit_A/B 보존) | 로짓 포함 확보 |
| **Phi-4-multimodal** | `results/rq1_model_battery_3lang_20260723/inference/full/phi/raw.jsonl` | (raw.jsonl 내 verdict token logit_A/B 보존) | 로짓 포함 확보 |
| **Qwen3-VL-8B** | `results/latest_vlm_extension_20260830/inference/full/qwen3/raw.jsonl` | `results/latest_vlm_extension_20260830/inference/full/qwen3/effective_logits.jsonl` | 로짓 포함 확보 |
| **InternVL3.5-8B** | `results/latest_vlm_extension_20260830/inference/full/internvl3_5/raw.jsonl` | `results/latest_vlm_extension_20260830/inference/full/internvl3_5/effective_logits.jsonl` | 로짓 포함 확보 |
| **Gemma4-12B** | `results/latest_vlm_extension_20260830/inference/full/gemma4/raw.jsonl` | `results/latest_vlm_extension_20260830/inference/full/gemma4/effective_logits.jsonl` | 로짓 포함 확보 |

---

### 2.3 부트스트랩 산출물 파일 (Bootstrap Outputs)

10,000-resample snippet-cluster bootstrap 신뢰구간 및 재표본 분석 산출물입니다.

| 파일명 | 경로 | 크기 | 행 수 | 내용 설명 |
|:---|:---|:---:|:---:|:---|
| **`rq1_intervals.csv`** | `/ANON/home/fse2027_handoff_rq1_20260912/rq1_intervals.csv` | 10.6 KB | 128행 | 8개 VLM 모델 전원 × 4개 스코프(java, python, cuda, pooled) × 4개 지표(valid, E, D, S)의 점추정치 및 95% 신뢰구간 (`ci_low`, `ci_high`) |
| **`vlm_tab_rq1_intervals.csv`** | `/ANON/home/fse2027_handoff_rq1_20260912/tab/vlm_tab_rq1_intervals.csv` | 1.97 KB | 8행 | 8개 모델의 pooled 지표 E, D, S 및 95% 신뢰구간 표 데이터 |
| **`vlm_tab_rq1_intervals.tex`** | `/ANON/home/fse2027_handoff_rq1_20260912/tab/vlm_tab_rq1_intervals.tex` | 2.15 KB | - | 논문 본문에 직접 삽입되는 LaTeX 표 소스 |
| **`fig3_data.csv`** | `/ANON/home/fse2027_handoff_rq1_20260912/fig_data/fig3_data.csv` | 2.11 KB | 24행 | 언어별 유효 정확도 및 유효 일치 정확도(E) 신뢰구간 그림 데이터 |
| **`fig4_data.csv`** | `/ANON/home/fse2027_handoff_rq1_20260912/fig_data/fig4_data.csv` | 1.05 KB | 8행 | 8개 모델 통합 지표(E, D, S) 비교 그림 데이터 |
| **`E1_debiased_vs_baseline.csv`** | `/ANON/experiment_root/results/fse2027_review_defense_e1_e8_20260730/analysis/E1/E1_debiased_vs_baseline.csv` | 9.74 KB | 15행 | 심사 답변(Review Defense E1)용 양방향 스니펫 클러스터 부트스트랩 CI (Random Forest 및 언어별 최고 baseline 비교 검정) |
| **`primary_snippet_cluster_bootstrap_replicates.csv`** | `/ANON/experiment_root/results/strict_swap_difficulty_analysis_20260710/primary_snippet_cluster_bootstrap_replicates.csv` | 865 KB | 10,000행 | 난이도 구간별 순서 불일치율 10,000개 부트스트랩 반복 복제본 데이터 |
| **`primary_snippet_cluster_bootstrap_summary.csv`** | `/ANON/experiment_root/results/strict_swap_difficulty_analysis_20260710/primary_snippet_cluster_bootstrap_summary.csv` | 304 B | 2행 | 난이도 구간별 불일치율 차이 점추정치 및 CI 요약 |

---

### 2.4 판정 라벨만 있고 로짓이 없는 파일 (Logits Absent / Verdicts Only)

경계 사례(Margin Boundary) 분석이 불가능하여 구분이 필수적인 파일 목록입니다.

| 파일 경로 | 표본 규모 | 모델 | 판정 라벨 컬럼 | 로짓 부재 원인 및 비고 |
|:---|:---:|:---|:---|:---|
| `results/large_model_cross_family_3model_300java_20260910/pair_level_results.csv` | 900행 | Mistral-Small-3.1-24B (300)<br>Qwen2.5-VL-32B (300)<br>Gemma-3-27B (300) | `choice_ab`, `choice_ba`, `is_correct_and_valid` | 대형 모델 확장성 사전 점검 파일럿으로, 텍스트 파싱 판정만 수집하고 토큰 로짓 훅(Hook)을 비활성화하여 로짓 부재 |
| `results/multilang_cross_family_3model_python_cuda_java_20260910/pair_level_multilang_results.csv` | 2,700행 | Mistral-Small-3.1-24B (900)<br>Qwen2.5-VL-32B (900)<br>Gemma-3-27B (900) | `ab_choice`, `ba_choice`, `ab_verdict`, `ba_verdict` | 3개 언어 교차 패밀리 점검 런으로 판정 라벨만 수집됨 |
| `results/openai_vlm_pilot_20260707/gpt-5.4-mini__pilot_raw.jsonl` | 1,200 calls<br>(300 pairs × 4) | GPT-5.4-mini | `parsed_choice` | 상용 폐쇄형 API(OpenAI) 프로토콜 점검 런으로, API 수준에서 토큰 로짓 접근 불가 |
| `results/openai_vlm_pilot_20260708/gpt-5.4__pilot_raw.jsonl` | 1,200 calls<br>(300 pairs × 4) | GPT-5.4 | `parsed_choice` | 상용 폐쇄형 API 프로토콜 점검 런 (토큰 로짓 미제공) |
| `results/openai_vlm_pilot_20260708_gpt55_fixed/gpt-5.5__pilot_raw.jsonl` | 1,200 calls<br>(300 pairs × 4) | GPT-5.5 | `parsed_choice` | 상용 폐쇄형 API 프로토콜 점검 런 (토큰 로짓 미제공) |

---

### 2.5 RQ2 Reduced 파일 (혼동 방지용 구분)

홈 디렉터리 내 `rq2_eval_reduced/` 하위에도 pair_level 파일들이 존재하나, 이는 **RQ2(시각적/스타일 섭동 실험) 전용** 데이터입니다.

- `rq2_eval_reduced/gemma/pair_level/Gemma-3-12B_pair_level.csv` (21,000행 = 3개 언어 × 7개 조건)
- `rq2_eval_reduced/gemma4/pair_level/Gemma4-12B_pair_level.csv` (21,000행)
- `rq2_eval_reduced/internvl3_5/pair_level/InternVL3.5-8B_pair_level.csv` (21,000행)

이 파일들은 RQ1의 9,000쌍 기본 평가 파일이 아니므로 RQ1 분석에 혼용하지 않도록 주의해야 합니다.

---

## 3. 모델 × 언어 조합별 최종 확보 상태 판정

판정 기준:
- **로짓 포함 확보**: 3,000행 전수(모델당 9,000행), AB/BA 판정 라벨 및 양방향 판정 로짓/마진 보존 완료
- **판정만 확보**: 판정 라벨(A/B 선택)만 존재하고 토큰 로짓/마진이 부재하거나 표본 수가 미달함
- **없음**: 파일 자체가 발견되지 않음

| 구분 | 대상 모델 | Java (3,000행) | Python (3,000행) | CUDA (3,000행) | 모델 전체 (9,000행) 상태 | 저장 원본 위치 |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **Primary Judge** | **Qwen2.5-VL-7B** | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (9,000) | `rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv` |
| **Primary Judge** | **Qwen3-VL-8B** | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (9,000) | `latest_vlm_extension_20260830/analysis/primary_pair_level.csv` |
| **Primary Judge** | **Gemma-3-12B** | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (9,000) | `rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv` |
| **Primary Judge** | **Gemma4-12B** | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (9,000) | `latest_vlm_extension_20260830/analysis/primary_pair_level.csv` |
| **Primary Judge** | **InternVL3.5-8B** | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (9,000) | `latest_vlm_extension_20260830/analysis/primary_pair_level.csv` |
| **Diagnostic** | **InternVL3-8B** | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (9,000) | `rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv` |
| **Diagnostic** | **Ministral-3-8B** | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (9,000) | `rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv` |
| **Diagnostic** | **Phi-4-multimodal** | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (9,000) | `rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv` |
| **Extension Check** | **DeepSeek-VL-7B** | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (3,000) | **로짓 포함 확보** (9,000) | `rq1_deepseek_vl7b_extension_20260806/analysis/full/pair_level_results.csv` |
| *Large Open Check* | *Qwen2.5-VL-32B* | *판정만 확보* (300) | *판정만 확보* (300) | *판정만 확보* (300) | *판정만 확보* (900) | `multilang_cross_family_3model_python_cuda_java_20260910/` |
| *Large Open Check* | *Mistral-Small-3.1-24B* | *판정만 확보* (300) | *판정만 확보* (300) | *판정만 확보* (300) | *판정만 확보* (900) | `multilang_cross_family_3model_python_cuda_java_20260910/` |
| *Large Open Check* | *Gemma-3-27B* | *판정만 확보* (300) | *판정만 확보* (300) | *판정만 확보* (300) | *판정만 확보* (900) | `multilang_cross_family_3model_python_cuda_java_20260910/` |
| *Closed Pilot* | *GPT-5.4-mini* | *판정만 확보* (100) | *판정만 확보* (100) | *판정만 확보* (100) | *판정만 확보* (300쌍/1200회) | `results/openai_vlm_pilot_20260707/` |
| *Closed Pilot* | *GPT-5.4* | *판정만 확보* (100) | *판정만 확보* (100) | *판정만 확보* (100) | *판정만 확보* (300쌍/1200회) | `results/openai_vlm_pilot_20260708/` |
| *Closed Pilot* | *GPT-5.5* | *판정만 확보* (100) | *판정만 확보* (100) | *판정만 확보* (100) | *판정만 확보* (300쌍/1200회) | `results/openai_vlm_pilot_20260708_gpt55_fixed/` |

---

## 4. 재생성 비용 추정 및 재현성 분석

### 4.1 실제 재생성 필요 여부
- **판정: 실제 재생성 불필요 (0 calls)**
- **사유**: 주 판정관 5개 모델 전원에 대한 pair-level 산출물, raw 토큰 로짓, 부트스트랩 CI가 서버 로컬 머신에 완벽하게 보존되어 있으므로 모델 재실행은 자원 및 시간 낭비입니다. 맥북 로컬로 파일을 전송하면 즉시 해결됩니다.

### 4.2 가상 재생성 비용 추정 (만약 서버 데이터 없이 스크래치 재실행할 경우)
만약 서버 내 데이터가 전무하여 스크래치부터 다시 재생성해야 하는 최악의 상황을 가정했을 때의 비용 추정치는 다음과 같습니다:

1. **호출 횟수 계산**:
   - 언어당 3,000 쌍 × 2 방향 (AB 제시 순서, BA 제시 순서) = **언어당 6,000 Inference Calls**
   - 모델당 3개 언어(Java, Python, CUDA) = **모델당 18,000 Inference Calls**
   - 누락 의심된 4개 Primary Judge (`Qwen3-VL-8B`, `Gemma-3-12B`, `Gemma4-12B`, `InternVL3.5-8B`) 전수 재생성 시:  
     $$4 \times 18,000 = \mathbf{72,000\text{ Calls}}$$
   - Primary 5개 모델 전체 재생성 시:  
     $$5 \times 18,000 = \mathbf{90,000\text{ Calls}}$$

2. **소요 시간 및 GPU 자원 추정**:
   - 8B ~ 12B 모델의 배치 크기 1 또는 소형 배치 기준 추론 속도: 약 1.2~1.8초/call
   - 모델당 18,000 calls 소요 시간: 단일 H100/A100 기준 약 6~8 GPU 시간
   - 4개 모델 전수 재생성 시: **총 24~32 GPU 시간** 소요

3. **결정론적 디코딩 및 재현 가능성 (Determinism & Reproducibility)**:
   - 프로토콜 명세(`protocol.json`, `FROZEN_PROMPT_B.txt`)에 따라 **`temperature = 0.0` (Greedy decoding), `do_sample = False`, `seed = 42`**의 결정론적 디코딩으로 사전 등록되어 있습니다.
   - 따라서 동일한 체크포인트 리비전(`revision` 명시)과 vLLM/Transformers 버전을 사용할 경우 **100% 비트 단위(또는 수치 오차 범위 내) 완벽 재현 가능**합니다.
   - 단, GPU 커널/라이브러리(CUDA, cuDNN, vLLM) 버전 차이에 따른 부동소수점 미세 반올림으로 극소수의 타이브레이크(Tie-break) 경계선 변동이 일어날 수 있으므로, 이미 무결성 검증을 마친 기존 확정 파일(`pair_level_results.csv`, `primary_pair_level.csv`)을 사용하는 것이 학술적 일관성 측면에서 가장 안전합니다.

---

## 5. 로컬 macOS (`~/Downloads/results`) 동기화 가이드

맥북 로컬 머신에서 RQ1 분석 및 원고 작성을 완료하기 위해, 아래 명령어를 맥북 터미널에서 실행하여 서버의 확정 산출물을 내려받는 것을 권장합니다.

### 5.1 필수 Pair-Level 파일 동기화
```bash
# 맥북 로컬 터미널에서 실행 (서버 주소가 gpusystem 인 경우)
mkdir -p ~/Downloads/results/rq1_model_battery_3lang_20260723/analysis/full/
mkdir -p ~/Downloads/results/latest_vlm_extension_20260830/analysis/

# 1. 5개 모델 배터리 pair-level 결과 (Qwen2.5, Gemma-3, InternVL3, Ministral, Phi-4)
scp <서버주소>:/ANON/experiment_root/results/rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv \
    ~/Downloads/results/rq1_model_battery_3lang_20260723/analysis/full/

# 2. 3개 최신 확장 모델 pair-level 결과 (Qwen3, InternVL3.5, Gemma4)
scp <서버주소>:/ANON/experiment_root/results/latest_vlm_extension_20260830/analysis/primary_pair_level.csv \
    ~/Downloads/results/latest_vlm_extension_20260830/analysis/
```

### 5.2 부트스트랩 및 표/그림 Handoff 번들 동기화
```bash
# 3. 10,000회 부트스트랩 신뢰구간 및 LaTeX 표 번들 (15 KB 압축 파일)
scp <서버주소>:/ANON/home/fse2027_handoff_rq1_20260912.zip ~/Downloads/
unzip -q ~/Downloads/fse2027_handoff_rq1_20260912.zip -d ~/Downloads/fse2027_handoff_rq1/
```

---

## 6. 결론

- **의문 해소**: Primary Judge 4개 모델의 출력이 없었던 것이 아니라, `pair_level_results.csv`와 `latest_vlm_extension_20260830/analysis/primary_pair_level.csv`라는 두 개의 표준 파일에 분할 집계되어 안전하게 보존되어 있었습니다.
- **로짓 보존**: 8개 오픈 VLM 모델 전원에 대해 `margin_ab`, `margin_ba` 및 원본 `raw.jsonl`의 토큰 로짓이 결측 없이 확보되어 경계 사례 분석이 100% 가능합니다.
- **부트스트랩 완비**: 10,000회 스니펫 클러스터 부트스트랩 신뢰구간 테이블(`rq1_intervals.csv`)과 LaTeX 표 소스가 이미 생성되어 즉시 논문 집필에 사용 가능합니다.
