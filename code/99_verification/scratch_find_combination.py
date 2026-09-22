import pandas as pd
import os

base_dirs = ["/ANON/experiment_root/fse2027/current/results"]

for root, dirs, files in os.walk(base_dirs[0]):
    for f in files:
        if f.endswith('.csv'):
            path = os.path.join(root, f)
            try:
                df = pd.read_csv(path)
                match_501 = False
                match_556 = False
                match_543 = False
                for col in df.select_dtypes(include=['float64', 'float32']):
                    if df[col].round(4).isin([0.501, 0.5010]).any(): match_501 = True
                    if df[col].round(4).isin([0.556, 0.5560]).any(): match_556 = True
                    if df[col].round(4).isin([0.543, 0.5433]).any(): match_543 = True
                
                if match_501 and match_556 and match_543:
                    print(f"BINGO! File containing ALL THREE values: {path}")
            except:
                pass
