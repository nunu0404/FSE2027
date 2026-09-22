> **Package note (added when this replication package was assembled).**
> This file is the original run README, kept as provenance. Two of its relative
> paths refer to the source tree, not to this package:
>
> | Original reference | Where it is here |
> | --- | --- |
> | `../qwen2.5_vl_7b_text_only_{ocr,source}.jsonl` | `../raw_inference/text_ablation/` |
> | `../backup_buggy_lang_20260920/` | **not shipped** — the pre-fix run, superseded by a full rerun on 2026-09-20. Its only use was the scope check described at the end of this file |
>
> The tie counts it records (source 164, ocr 280 exact `c = 0` pairs excluded)
> are what convert this directory's `debiased_accuracy` column to the
> manuscript's Table 3(c) `D` values of 56.30% and 52.58%, which count ties as
> errors over all 9,000 pairs. See `CLAIMS_TO_ARTIFACTS.md`, Table 3(c).
>
> The body below is in Korean, as written.

---

# text_ablation_20260920 — 재계산 분석 산출물

생성: 2026-09-21 / 원본: `../qwen2.5_vl_7b_text_only_{ocr,source}.jsonl`
실행: Qwen2.5-VL-7B-Instruct, 텍스트 전용, 9,000 pairs x 2 orders, GPU0
(`run_all.sh`, `CUDA_VISIBLE_DEVICES=0`), 2026-09-20 15:02 정상 종료.

## 파일

| 파일 | 내용 |
|---|---|
| `pair_level_text_only_source.csv` | source 조건 pair-level 재계산 (9,000행) |
| `pair_level_text_only_ocr.csv` | OCR 조건 pair-level 재계산 (9,000행) |
| `summary_metrics.csv` | 조건 x (overall/language/difficulty) 지표, Wilson 95% CI 포함 |
| `paired_contrasts.csv` | 쌍체 McNemar 대조 |

## 주의: 원본 jsonl의 승계 필드

`run_vlm_text_ablation.py`가 `result = pair.copy()`로 결과를 만들기 때문에,
원본 jsonl의 `is_valid_strict_swap` / `is_correct` / `parse_failures` /
`parsed_ab` / `parsed_ba` / `preference_ab` / `preference_ba` 는
**이번 실행 결과가 아니라 입력 pair 파일(image_only promptB seed42 실행분)의 값**이다.

이 디렉터리의 pair-level CSV는 전부 이번 실행의
`parsed_choice` / `logit_A` / `logit_B` 로 재계산한 값이다.
승계 필드는 `base_valid` / `base_correct` / `base_parsefail` 로 이름을 바꿔
image_only 기준선으로만 분리해 두었다.

## 지표 정의

- `strict_valid` = 양쪽 order 파싱 성공 AND 선택 스니펫 일치
- `E` (effective) = 정답 pair / 전체 pair
- `V` (valid) = 정답 pair / strict_valid pair
- `D` (debiased) = c_content = (m_ab - m_ba)/2 부호로 판정, m = logit_A - logit_B.
  정확 동점(c=0)은 제외 (source 164, ocr 280).
- `S` (strict-swap error) = 1 - strict_valid 비율
  = `parse_failure_rate` + `order_disagreement_rate`.
  이번 실행은 두 조건 모두 파싱 실패 0건이므로 S 전체가 순서 불일치.

## 언어 라벨 버그 재실행 범위 검증

`../backup_buggy_lang_20260920/` 와 대조한 결과:
java 6,000콜 결과 동일(변경 0), python/cuda 12,000콜 중
10,341(ocr) / 10,735(source)콜의 출력이 변경됨. 재실행 범위 정상.
