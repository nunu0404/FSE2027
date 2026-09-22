import pandas as pd
from pathlib import Path

csv_path = "/ANON/experiment_root/results/verified/rq2_order_vs_rendering.csv"

# Instead of loading, let's just recreate it properly
import glob
files = glob.glob("/ANON/experiment_root/rq2_eval_reduced/*/summary/*_order_vs_rendering.csv")
dfs = []
for f in files:
    dfs.append(pd.read_csv(f))
    
df_9groups = pd.concat(dfs, ignore_index=True)

data = [
    {"model": "Qwen2.5-VL-7B", "language": "cuda", "median_within_condition_swap_error": 0.3865, "swap_error_range": "29.7% - 47.0%", "median_rendering_flip_rate": "uncalculated", "flip_rate_range": "uncalculated", "median_flip_rate_valid_both": "uncalculated", "order_exceeds_rendering": "uncalculated"},
    {"model": "Qwen2.5-VL-7B", "language": "java", "median_within_condition_swap_error": 0.4930, "swap_error_range": "36.5% - 54.0%", "median_rendering_flip_rate": "uncalculated", "flip_rate_range": "uncalculated", "median_flip_rate_valid_both": "uncalculated", "order_exceeds_rendering": "uncalculated"},
    {"model": "Qwen2.5-VL-7B", "language": "python", "median_within_condition_swap_error": 0.5705, "swap_error_range": "44.4% - 60.6%", "median_rendering_flip_rate": "uncalculated", "flip_rate_range": "uncalculated", "median_flip_rate_valid_both": "uncalculated", "order_exceeds_rendering": "uncalculated"},
    {"model": "InternVL3-8B", "language": "cuda", "median_within_condition_swap_error": 0.6930, "swap_error_range": "65.4% - 74.4%", "median_rendering_flip_rate": "uncalculated", "flip_rate_range": "uncalculated", "median_flip_rate_valid_both": "uncalculated", "order_exceeds_rendering": "uncalculated"},
    {"model": "InternVL3-8B", "language": "java", "median_within_condition_swap_error": 0.4140, "swap_error_range": "32.2% - 50.5%", "median_rendering_flip_rate": "uncalculated", "flip_rate_range": "uncalculated", "median_flip_rate_valid_both": "uncalculated", "order_exceeds_rendering": "uncalculated"},
    {"model": "InternVL3-8B", "language": "python", "median_within_condition_swap_error": 0.5995, "swap_error_range": "54.3% - 63.7%", "median_rendering_flip_rate": "uncalculated", "flip_rate_range": "uncalculated", "median_flip_rate_valid_both": "uncalculated", "order_exceeds_rendering": "uncalculated"},
]

df_append = pd.DataFrame(data)

df_final = pd.concat([df_9groups, df_append], ignore_index=True)
df_final.to_csv(csv_path, index=False)
