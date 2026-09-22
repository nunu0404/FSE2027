import pandas as pd
import numpy as np

print("=== Task 8: Baseline Selection Bias (9 Classical Feature Baselines) ===")

# Load ML predictions / summary
p_ml_sum = "results/rq1_model_battery_3lang_20260723/analysis/table1_ml_replication/canonical_pairwise_summary.csv"
df_sum = pd.read_csv(p_ml_sum)

# Inspect unique models in summary:
print("Models in canonical_pairwise_summary:", df_sum["model"].unique().tolist())
print(df_sum[["model", "language", "effective_accuracy"]].head(10))

# Create pivot table of Model x Language:
pivot = df_sum.pivot(index="model", columns="language", values="effective_accuracy") * 100
# Ensure language column names are lowercase
pivot.columns = [c.lower() for c in pivot.columns]

# Reorder columns: cuda, java, python
pivot = pivot[["cuda", "java", "python"]]
pivot["pooled_3lang"] = pivot.mean(axis=1)

# Add Summary Statistics: Mean and Median across 9 models
mean_row = pivot.mean(axis=0)
median_row = pivot.median(axis=0)

pivot.loc["MEAN (Expected Value across 9 models)"] = mean_row
pivot.loc["MEDIAN across 9 models"] = median_row

print("\n=== Table: 9 Source Feature Baselines by Language (Accuracy %) ===")
print(pivot.round(2).to_string())

