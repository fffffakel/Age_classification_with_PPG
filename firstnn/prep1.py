import pandas as pd
import re

heartpy_metrics = pd.read_csv(r'D:\Proga\AML\preprocessed_data\summary.csv')
subject_info = pd.read_csv(r'D:\Proga\AML\panacea\subject-info.csv')

def parse_metrics(metrics_str):
    if 'Could not determine best fit' in metrics_str:
        return None
    
    pattern = r"'(\w+)':\s*(np\.float64\(|)([\d.]+)(\)|)"
    matches = re.findall(pattern, metrics_str)
    
    if not matches:
        return None
    
    metrics_dict = {}
    for key, _, value, _ in matches:
        value = float(value)
        metrics_dict[key] = value
    
    return metrics_dict

heartpy_metrics['HeartPy_Metrics'] = heartpy_metrics['HeartPy_Metrics'].apply(parse_metrics)

heartpy_metrics = heartpy_metrics.dropna(subset=['HeartPy_Metrics'])

merged_data = pd.merge(subject_info, heartpy_metrics, on='ID')

result = []
for index, row in merged_data.iterrows():
    age_group = row['Age_group']
    metrics = row['HeartPy_Metrics']
    result.append([age_group, metrics])

print(result[0])