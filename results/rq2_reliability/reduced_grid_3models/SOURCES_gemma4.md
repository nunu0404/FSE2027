# SOURCES.md: FSE2027 RQ2 Reduced Grid Gemma4-12B Handoff Bundle

> **생성 일시**: 2026-09-12 (KST)
> **대상 모델**: `google/gemma-4-12B-it` (리비전: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`)
> **Logit Mismatch Note**: Across 42,000 calls, 135 (0.32%) exhibited logit argmax vs parsed verdict disagreement: 101 cases were bfloat16 exact logit ties (|margin| = 0.0000) where greedy decoding picked token ' A' (id 562) while argmax logic reported TIE, and 34 cases were non-conforming responses where verdict regex failed to match (parse failures); neither category alters strict-swap validity or correctness computed from parsed text.

## 1. 파일 목록 및 SHA-256 체크섬

| 파일 경로 | 크기 (Bytes) | 행 수 (Lines) | SHA-256 체크섬 | 설명 |
| :--- | :---: | :---: | :--- | :--- |
| `figures_input/Gemma4-12B_fig6_input.csv` | 477 | 4 | `70285b543c802f9ff62f5bff2b092bb0012ee43a20d24c32d788ee3af8391013` | |
| `manifests/gemma4_12b_reduced_manifest.json` | 352 | 12 | `58dab3984406ff38a45b2cc8db07c99619bac5fc9a6215995a6ab756b5ee7231` | |
| `summary/Gemma4-12B_expectations.json` | 132 | 5 | `5b5414d7fd83ca0d56b35167c915d62f48dddd706419b66833544ebff850bb0f` | |
| `summary/Gemma4-12B_matched_contrasts.csv` | 4694 | 19 | `29750ce1998110b3e53ca530b238ac069aa8bea480851896a98ba419704c1580` | |
| `summary/Gemma4-12B_order_vs_rendering.csv` | 477 | 4 | `70285b543c802f9ff62f5bff2b092bb0012ee43a20d24c32d788ee3af8391013` | |
| `summary/Gemma4-12B_summary_metrics.csv` | 4723 | 22 | `a0147c818935c2b920de56e6205c6e4678caa9dc45fcff9bf6aa1122f768436a` | |
