import pandas as pd
import os

base_dirs = [
    "/ANON/experiment_root/fse2027/current/results"
]

target_values = [0.501, 0.5010, 0.556, 0.5560, 0.5433, 0.543]

for root, dirs, files in os.walk(base_dirs[0]):
    for f in files:
        if f.endswith('.csv'):
            path = os.path.join(root, f)
            try:
                df = pd.read_csv(path)
                for col in df.select_dtypes(include=['float64', 'float32']):
                    # Check if any value rounded to 3 or 4 decimals matches
                    matches = df[df[col].round(4).isin([0.501, 0.5010, 0.556, 0.5560, 0.5433, 0.543])]
                    if not matches.empty:
                        print(f"Match found in {path}, column {col}")
                        print(matches[[c for c in df.columns if c == col or df[c].dtype == 'object']].head(3))
            except Exception as e:
                pass
