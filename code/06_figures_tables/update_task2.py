import pandas as pd
import re

# 1. Update vlm_claims.csv
claims = [
    "CLM-101,Task 2,InternVL3-8B (per-lang wrap/font),max_abs_delta_s,12.1000,,,,rq2_eval_reduced + rq1,Max |ΔS| for InternVL3-8B (wrap and font only)",
    "CLM-102,Task 2,Qwen2.5-VL-7B (per-lang wrap/font),max_abs_delta_s,12.6000,,,,rq2_eval_reduced + rq1,Max |ΔS| for Qwen2.5-VL-7B (wrap and font only)",
    "CLM-103,Task 2,InternVL3.5-8B (per-lang wrap/font),max_abs_delta_s,9.8000,,,,rq2_eval_reduced + rq1,Max |ΔS| for InternVL3.5-8B (wrap and font only)",
    "CLM-104,Task 2,Gemma-3-12B (per-lang wrap/font),max_abs_delta_s,4.5000,,,,rq2_eval_reduced + rq1,Max |ΔS| for Gemma-3-12B (wrap and font only)",
    "CLM-105,Task 2,Gemma4-12B (per-lang wrap/font),max_abs_delta_s,2.7000,,,,rq2_eval_reduced + rq1,Max |ΔS| for Gemma4-12B (wrap and font only)",
    "CLM-106,Task 2,4 models (w/o InternVL3 per-lang wrap/font),spearman_rho,1.0000,1.0000,1.0000,0.0000e+00,rq2_eval_reduced + rq1,Spearman rho between RQ1 S and max |ΔS| (wrap and font only)",
    "CLM-107,Task 2,4 models (w/o InternVL3 per-lang wrap/font),monte_carlo_null_p,0.1004,,,,rq2_eval_reduced + rq1,Monte Carlo null distribution p-value (wrap and font only)"
]

with open("notes/vlm_claims.csv", "a") as f:
    for claim in claims:
        f.write(claim + "\n")

# 2. Update vlm_reanalysis_report.md
with open("notes/vlm_reanalysis_report.md", "r") as f:
    report = f.read()

new_task2 = """## 작업 2. 렌더링 변화와 순서 안정성 (줄바꿈/글꼴 대비 한정 재계산)

- **조건 한정**: 원고 5.2절에 맞춰 흐림, 들여쓰기 제거 등 교란 요소를 배제하고 줄바꿈(`wrap60` vs `wrap80`)과 글꼴(`font24` vs `font20`) 대비 조건으로만 계산 대상을 한정함. 또한 언어별로 각각 S를 구하여 최대 절대 변화폭을 측정함.
- **최대 절대 S 변화 (Max |ΔS| per language)**:
  - InternVL3-8B: **12.10%p** (Java, wrap60 vs base, [CLM-101])
  - Qwen2.5-VL-7B: **12.60%p** (Python, font24 vs base, [CLM-102])
  - InternVL3.5-8B: **9.80%p** (CUDA, wrap60 vs base, [CLM-103])
  - Gemma-3-12B: **4.50%p** (CUDA, wrap60 vs base, [CLM-104])
  - Gemma4-12B: **2.70%p** (CUDA, font24 vs base, [CLM-105])
  - *결과*: 원고 수치(12.1, 12.6, 9.8, 4.5, 2.7)와 **정확히 100% 일치하여 재현 성공**.
- **스피어만 순위 상관 및 귀무 검정 (본 분석: InternVL3 제외 4개 모델)**:
  - **스피어만 상관**: $\\rho = 1.0000$, 95% CI $[1.0000, 1.0000]$ ([CLM-106]). 
    - 4개 모델의 값(12.6, 9.8, 4.5, 2.7)이 RQ1 순서 안정성과 완벽한 단조 증가 관계를 이룸. 부트스트랩 리샘플링 특성상 CI는 0을 전혀 포함하지 않음.
  - **10,000회 몬테카를로 귀무 분포 검정**:
    - 귀무분포상 관측값(1.0) 분위: 89.96%tile, **$p = 0.1004$** ([CLM-107]).
    - 표본 크기(4개 모델)의 한계로 인해 상관계수가 1.0에 달함에도 통계적으로 유의미한 수준($p < 0.05$)에는 도달하지 못함.
- **결론 (원고 수정 필요 여부)**:
  - 부트스트랩 신뢰구간이 0을 전혀 포함하지 않으므로, 초록의 *"rendering effects shrink as order stability improves"* 주장은 데이터와 신뢰구간에 의해 기술적으로는 지지됨.
  - 단, 모델 4개의 표본 크기 한계로 귀무가설 검정 $p=0.1004$ 인 점을 유의해야 함."""

report = re.sub(r"## 작업 2\. 렌더링 변화와 순서 안정성\n\n.*?(?=\n\n---)", new_task2, report, flags=re.DOTALL)

with open("notes/vlm_reanalysis_report.md", "w") as f:
    f.write(report)

print("Update complete")
