import pandas as pd

source = pd.read_csv("/ANON/experiment_root/fse2027/current/results/python_cuda_missing_experiments_20260716/tables/clean_classical_results_by_language.csv")
ocr = pd.read_csv("/ANON/experiment_root/fse2027/current/results/python_cuda_missing_experiments_20260716/ocr/ocr_classical_results_by_language.csv")
java_source = pd.read_csv("/ANON/experiment_root/fse2027/current/results/screenshot_only_ocr_clean_20260707/clean_classical_results_by_language.csv")
java_ocr = pd.read_csv("/ANON/experiment_root/fse2027/current/results/screenshot_only_ocr_clean_20260707/ocr_classical_results_by_language.csv")

print("=== Python ===")
print("Source:", source[source['language'] == 'python'][['model', 'pair_accuracy']].to_string(index=False))
print("OCR Rapid:", ocr[(ocr['language'] == 'python') & (ocr['ocr_engine'] == 'rapidocr')][['model', 'pair_accuracy']].to_string(index=False))
print("OCR Easy:", ocr[(ocr['language'] == 'python') & (ocr['ocr_engine'] == 'easyocr')][['model', 'pair_accuracy']].to_string(index=False))

print("\n=== CUDA ===")
print("Source:", source[source['language'] == 'cuda'][['model', 'pair_accuracy']].to_string(index=False))
print("OCR Rapid:", ocr[(ocr['language'] == 'cuda') & (ocr['ocr_engine'] == 'rapidocr')][['model', 'pair_accuracy']].to_string(index=False))
print("OCR Easy:", ocr[(ocr['language'] == 'cuda') & (ocr['ocr_engine'] == 'easyocr')][['model', 'pair_accuracy']].to_string(index=False))

print("\n=== Java ===")
print("Source:", java_source[java_source['language'] == 'java'][['model_family', 'pair_accuracy']].to_string(index=False))
print("OCR Rapid:", java_ocr[(java_ocr['language'] == 'java') & (java_ocr['ocr_engine'] == 'rapidocr')][['model', 'pair_accuracy']].to_string(index=False))
print("OCR Easy:", java_ocr[(java_ocr['language'] == 'java') & (java_ocr['ocr_engine'] == 'easyocr')][['model', 'pair_accuracy']].to_string(index=False))

