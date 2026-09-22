import pandas as pd
import os

base_dirs = [
    "/ANON/experiment_root/fse2027/current/results/python_cuda_missing_experiments_20260716/tables",
    "/ANON/experiment_root/fse2027/current/results/screenshot_only_ocr_clean_20260707"
]

for d in base_dirs:
    for f in os.listdir(d):
        if f.endswith('.csv'):
            path = os.path.join(d, f)
            try:
                df = pd.read_csv(path)
                for col in df.select_dtypes(include=['float64', 'float32']):
                    if any((df[col] >= 0.500) & (df[col] <= 0.56)):
                        # Look for something close to 50.10, 55.60, 54.33
                        matches = df[(df[col] >= 0.5009) & (df[col] <= 0.5011) | 
                                     (df[col] >= 0.5559) & (df[col] <= 0.5561) |
                                     (df[col] >= 0.5432) & (df[col] <= 0.5434)]
                        if not matches.empty:
                            print(f"Match found in {path}, column {col}")
                            print(matches)
            except Exception as e:
                pass
