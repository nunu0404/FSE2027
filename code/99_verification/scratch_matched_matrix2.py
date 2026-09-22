import pandas as pd

source = pd.read_csv("/ANON/experiment_root/fse2027/current/results/python_cuda_missing_experiments_20260716/tables/clean_classical_results_by_language.csv")
ocr = pd.read_csv("/ANON/experiment_root/fse2027/current/results/python_cuda_missing_experiments_20260716/ocr/ocr_classical_results_by_language.csv")

print("=== Python ===")
print("Source:")
print(source[source['language'] == 'python'][['model_family', 'pair_accuracy']].to_string(index=False))
print("OCR Rapid:")
print(ocr[(ocr['language'] == 'python') & (ocr['ocr_engine'] == 'rapidocr')][['model', 'pair_accuracy']].to_string(index=False))
print("OCR Easy:")
print(ocr[(ocr['language'] == 'python') & (ocr['ocr_engine'] == 'easyocr')][['model', 'pair_accuracy']].to_string(index=False))

print("\n=== CUDA ===")
print("Source:")
print(source[source['language'] == 'cuda'][['model_family', 'pair_accuracy']].to_string(index=False))
print("OCR Rapid:")
print(ocr[(ocr['language'] == 'cuda') & (ocr['ocr_engine'] == 'rapidocr')][['model', 'pair_accuracy']].to_string(index=False))
print("OCR Easy:")
print(ocr[(ocr['language'] == 'cuda') & (ocr['ocr_engine'] == 'easyocr')][['model', 'pair_accuracy']].to_string(index=False))
