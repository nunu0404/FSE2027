# SOURCES.md: FSE2027 RQ2 Reduced Grid Experiment Handoff Bundle

> **생성 일시**: 2026-09-12 02:54 (KST)
> **목적**: 원고 작성 에이전트가 서버나 원본 데이터베이스 접근 없이 RQ2 축소 그리드 결과를 논문(Section 5.2, Section 6.1, Figure 6 등)에 즉시 반영할 수 있도록 정제된 요약 데이터, 통계 검정, 입력 데이터셋, 매니페스트 및 보고서를 독립 패키지로 제공.

---

## 1. 판정자(Judge) 식별 및 메타데이터

| 논문 표기 | 모델 식별자 (HF Repo ID) | 리비전 해시 (Revision Hash) | 평가 호출 수 | 역할 |
| :--- | :--- | :--- | :---: | :--- |
| **Gemma-3-12B** | `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | 42,000 calls | third judge; RQ1 strict-swap error 21.8 percent |
| **InternVL3.5-8B** | `OpenGVLab/InternVL3_5-8B-HF` | `741a7d03020411e666c6109218ab71e08151ef86` | 42,000 calls | 최신 멀티모달 확장 판정자 |
| **Qwen2.5-VL-7B** | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | 6,000 calls | Baseline Anchor 재실행 (환경 재현성 검증) |

### 주의사항 (Gemma 식별):
- 본 축소 그리드 실행 체크포인트는 `google/gemma-3-12b-it` (리비전 `96b6f1e...`)입니다.
- 따라서 본 데이터는 **Gemma-3-12B**로 표기해야 하며 Gemma4-12B로 표기해서는 안 됩니다.
- 진짜 Gemma4-12B (`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`)는 로컬 스냅샷에 보관되어 있으며, 1x Blackwell GPU(96GB) 기준 약 6시간 45분이 소요됩니다.

---

## 2. 번들 내 파일 인덱스 및 SHA-256 해시

| 파일 경로 | 크기 (Bytes) | 행 수 (Lines) | SHA-256 체크섬 | 설명 |
| :--- | :---: | :---: | :--- | :--- |
| `REPORT_RQ2_REDUCED.md` | 14649 | 174 | `d2a8374638e58e4a078e3e0bf2acf9d1e8898de7714cb8eafe6241dad996353a` | 최종 분석 보고서 (수정본) |
| `anchor_qwen/qwen25_anchor_vs_original.csv` | 775 | 4 | `4b144b79fee480a7c526ad65c9d48de7838028ab8dfb073c0cc785a1753d6d9e` | Qwen2.5 Baseline Anchor 재실행 vs 기존 런 비교 대조표 |
| `gemma/figures_input/Gemma-3-12B_fig6_input.csv` | 413 | 4 | `e4e8c658f60408211101e3e803926048d961d4f34198b8712b45effb73d6bef9` | Figure 6(a) CDF/Scatter 플롯 입력 데이터 |
| `gemma/summary/Gemma-3-12B_expectations.json` | 131 | 5 | `32157782093389142346a6526e69a316f59de55c3165fe36747bb780d5adc4a8` | 사전 가설 E1~E3 성립 여부 JSON |
| `gemma/summary/Gemma-3-12B_matched_contrasts.csv` | 4475 | 19 | `dfaceb1b270f1a4b55b06b956ff690298f9106cc7470aa797b73b7054ddee003` | Baseline 대비 McNemar 및 Holm 보정 통계 검정 |
| `gemma/summary/Gemma-3-12B_order_vs_rendering.csv` | 413 | 4 | `e4e8c658f60408211101e3e803926048d961d4f34198b8712b45effb73d6bef9` | Section 6.1 복제: Swap Error vs Rendering Flip Rate |
| `gemma/summary/Gemma-3-12B_summary_metrics.csv` | 4725 | 22 | `f0649622cd19e38a10dc5f54b2edd52de40c5722ad1a7ff1dcd147f12605d868` | Section 5.2 복제: 조건별 Valid/Effective Acc, Swap Error 등 |
| `internvl3_5/figures_input/InternVL3.5-8B_fig6_input.csv` | 461 | 4 | `d7329c7116642392dfb939fe33e5b425eb6f2513623f8ec23a98b385d515abfd` | Figure 6(a) CDF/Scatter 플롯 입력 데이터 |
| `internvl3_5/summary/InternVL3.5-8B_expectations.json` | 131 | 5 | `32157782093389142346a6526e69a316f59de55c3165fe36747bb780d5adc4a8` | 사전 가설 E1~E3 성립 여부 JSON |
| `internvl3_5/summary/InternVL3.5-8B_matched_contrasts.csv` | 4688 | 19 | `ca2fc9400599349c36bfcc5ea94022bebeea2ef0b9da4311593d6f8df495ca3d` | Baseline 대비 McNemar 및 Holm 보정 통계 검정 |
| `internvl3_5/summary/InternVL3.5-8B_order_vs_rendering.csv` | 461 | 4 | `d7329c7116642392dfb939fe33e5b425eb6f2513623f8ec23a98b385d515abfd` | Section 6.1 복제: Swap Error vs Rendering Flip Rate |
| `internvl3_5/summary/InternVL3.5-8B_summary_metrics.csv` | 4787 | 22 | `733a41831b79f0ded2e4aaac764da8bef909437473816f1bc7b0f6d35d9cba6a` | Section 5.2 복제: 조건별 Valid/Effective Acc, Swap Error 등 |
| `manifests/gemma_3_12b_reduced_manifest.json` | 353 | 12 | `3e91078edea67a2dae0a07efdff90f3f076e9724f4a3e930e1e57196efee85a5` | 추론 무결성 및 환경 매니페스트 |
| `manifests/internvl3_5_8b_reduced_manifest.json` | 363 | 12 | `ec2a45f76c01d5f0a94b8c2b44440533aa71fc21199d570d3560b5f0074040da` | 추론 무결성 및 환경 매니페스트 |
| `manifests/qwen25_vl_7b_anchor_manifest.json` | 326 | 11 | `a50bd75a54b42895d7c1d8711d9ef984b39f483c1fbc814283b3234b2ff4d293` | Qwen2.5 Baseline Anchor 재실행 vs 기존 런 비교 대조표 |

---

## 3. 핵심 수치 빠른 참조 (원고 삽입용)

### 3.1. 순서 교란 vs 렌더링 교란 (Section 6.1)
- **Gemma-3-12B**:
  - CUDA: Swap Error 중앙값 18.95% vs 렌더링 뒤집힘 중앙값 18.25% (차이: **+0.70%p, 3 points 미만**), Both-valid 뒤집힘 7.68%
  - Java: Swap Error 중앙값 18.30% vs 렌더링 뒤집힘 중앙값 15.65% (차이: **+2.65%p, 3 points 미만**), Both-valid 뒤집힘 5.84%
  - Python: Swap Error 중앙값 27.80% vs 렌더링 뒤집힘 중앙값 19.25% (차이: **+8.55%p**), Both-valid 뒤집힘 8.74%
- **InternVL3.5-8B**:
  - CUDA: Swap Error 중앙값 29.35% vs 렌더링 뒤집힘 중앙값 13.25% (차이: **+16.10%p**), Both-valid 뒤집힘 3.82%
  - Java: Swap Error 중앙값 30.25% vs 렌더링 뒤집힘 중앙값 15.92% (차이: **+14.33%p**), Both-valid 뒤집힘 5.70%
  - Python: Swap Error 중앙값 37.70% vs 렌더링 뒤집힘 중앙값 14.11% (차이: **+23.59%p**), Both-valid 뒤집힘 3.14%

### 3.2. InternVL3.5 247건 미파싱 처리
- 247건(0.59%)은 드롭(drop)되지 않고 쌍 평가에서 `parsed_both = False`로 분류되어 유효하지 않은 쌍(invalid pairs)으로 E($n_{correct}/1000$) 및 S($1 - n_{valid}/1000$) 분모에 완전히 포함됨.

### 3.3. Qwen Anchor 재현성 두 줄 진술
> 동일 환경 재실행(same-environment re-run) 결과는 모든 평가 지표에서 최대 0.8%p 차이 이내로 나타났습니다.  
> 원고에 보고된 3.70%p 격차는 서로 다른 렌더러 및 실행 환경 간의 차이로 인한 것입니다.
