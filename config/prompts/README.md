# Judging prompt

`prompt_B_readability_judge.txt` is the single frozen judging instruction used by
every open-weight and closed VLM run in the paper (Figure 3 in the manuscript).
`{language}` is substituted with `Java`, `Python`, or `CUDA`.

The three separately frozen copies kept by the original runs
(`grounded_protocol_3lang_20260721/config/prompt_B_template.txt`,
`latest_vlm_extension_20260830/config/FROZEN_PROMPT_B.txt`,
`deploy_v2_logit_20260731/config/F5_FROZEN_PROMPT_B.txt`) are byte-identical:

    SHA-256 1eb486ba7a7f1908...  (see MANIFEST.sha256 for the full digest)

Model-specific chat templates and image layouts are applied by the runner
scripts in `code/02_inference/`, not by this file.
