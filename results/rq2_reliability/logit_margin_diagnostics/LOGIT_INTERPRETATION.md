# Logit decomposition interpretation

## Korean

AB/BA verdict margin을 위치 성분 b와 내용 성분 c로 분해했다. 비경계 행에서는 `|c|>|b|`와 반대 margin 부호의 관계가 전수 성립했다. BF16 A/B logit 동률 경계는 별도로 보고하며, sign(c) 기반 정확도는 위치 성분을 대수적으로 제거한 two-pass 추정치이지 관측된 strict-valid 판정으로 재분류한 값이 아니다.

## English

We decomposed AB/BA verdict margins into a position component b and a content component c. The `|c|>|b|` identity held for every non-boundary row. Exact BF16 A/B-logit ties are reported separately. Sign(c) accuracy is a two-pass position-debiased estimate, not a relabeling of observed swap-invalid pairs as strict-valid.
