import pandas as pd
import numpy as np
import scipy.stats as stats

source_df = pd.read_csv("/ANON/experiment_root/fse2027/current/results/python_cuda_missing_experiments_20260716/data/clean_classical_pair_predictions.csv")
ocr_df = pd.read_csv("/ANON/experiment_root/fse2027/current/results/python_cuda_missing_experiments_20260716/ocr/ocr_classical_pair_predictions.csv")

def get_ci(lang, source_model, ocr_engine, ocr_model):
    s = source_df[(source_df['language'] == lang) & (source_df['model'].str.contains(source_model))].copy()
    s = s.set_index('pair_id')
    
    o = ocr_df[(ocr_df['language'] == lang) & (ocr_df['ocr_engine'] == ocr_engine) & (ocr_df['model'].str.contains(ocr_model))].copy()
    o = o.set_index('pair_id')
    
    merged = s.join(o, lsuffix='_src', rsuffix='_ocr', how='inner')
    
    n_pairs = len(merged)
    
    merged['is_correct_src'] = merged['is_correct_src'].astype(bool)
    merged['is_correct_ocr'] = merged['is_correct_ocr'].astype(bool)
    
    acc_src = merged['is_correct_src'].mean()
    acc_ocr = merged['is_correct_ocr'].mean()
    delta = acc_ocr - acc_src
    
    n10 = sum(merged['is_correct_src'] & ~merged['is_correct_ocr'])
    n01 = sum(~merged['is_correct_src'] & merged['is_correct_ocr'])
    
    # Asymptotic standard error for paired differences
    se = np.sqrt( (n10 + n01 - ((n01 - n10)**2 / n_pairs)) ) / n_pairs
    z = stats.norm.ppf(0.975)
    margin = z * se
    
    ci_low = delta - margin
    ci_high = delta + margin
    
    print(f"[{lang.upper()}] Source {source_model} vs {ocr_engine} {ocr_model}")
    print(f"  Source Acc: {acc_src*100:.2f}% | OCR Acc: {acc_ocr*100:.2f}% | Delta: {delta*100:.2f}%p")
    print(f"  n10 (Src only): {n10}, n01 (OCR only): {n01}")
    print(f"  95% CI: [{ci_low*100:.2f}%p, {ci_high*100:.2f}%p]\n")


# 1. Highest Engine matched
get_ci('python', 'gradient_boosting_regressor', 'rapidocr', 'gradient_boosting_regressor')
get_ci('cuda', 'linear_regression', 'easyocr', 'linear_regression')

# 2. RF Matched (for RapidOCR)
get_ci('python', 'random_forest_regressor', 'rapidocr', 'random_forest_regressor')
get_ci('cuda', 'random_forest_regressor', 'rapidocr', 'random_forest_regressor')

get_ci('java', 'random_forest_regressor', 'rapidocr', 'random_forest_regressor')
get_ci('python', 'random_forest_regressor', 'rapidocr', 'random_forest_regressor')
