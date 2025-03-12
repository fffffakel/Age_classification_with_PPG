import os
import wfdb
import heartpy as hp
import pandas as pd
import numpy as np

os.chdir("D:\\Proga\\AML")

try:
    os.makedirs('preprocessed_data')
except FileExistsError:
    print('directory \"preprocessed_data\" already exists')

subjects = (pd.read_csv("panacea\\subject-info.csv", dtype={'ID': str}))[['ID', 'Age_group']]
subjects = subjects.dropna()
subjects.to_csv('preprocessed_data\\description.csv')

ids = subjects['ID'].values
Ages = subjects['Age_group'].values

result = []

for id in ids:
    try:
        record = wfdb.rdrecord('panacea\\' + id)

        channel_index = record.sig_name.index('NIBP')
        signal = record.p_signal[:, channel_index]
        fs = record.fs

        try:
            wd, m = hp.process(signal, fs)
            peaks = np.array([peak for peak in wd['peaklist'] if peak not in wd['removed_beats']])
            result.append([id, m])
            os.makedirs('preprocessed_data\\' + id, exist_ok=True)

            with open('preprocessed_data\\' + id + '\\peaks.csv', 'w') as file:
                for peak in peaks:
                    file.write(f"{peak}\n")

            metrics_df = pd.DataFrame(m.items(), columns=['Metric', 'Value'])
            metrics_df.to_csv('preprocessed_data\\' + id + '\\metrics.csv', index=False)
        except hp.exceptions.BadSignalWarning as e:
            print(f"Skipping ID {id} due to bad signal: {e}")
            result.append([id, f"BadSignalWarning: {e}"])
        except Exception as e:
            print(f"An error occurred for ID {id} during heartpy processing: {e}")
            result.append([id, f"Error during heartpy processing: {e}"])
    except FileNotFoundError as e:
        print(f"File not found for ID {id}: {e}")
        result.append([id, f"FileNotFoundError: {e}"])
    except Exception as e:
        print(f"An error occurred for ID {id} during record reading: {e}")
        result.append([id, f"Error during record reading: {e}"])


result_df = pd.DataFrame(result, columns=['ID', 'HeartPy_Metrics'])
result_df.to_csv('preprocessed_data\\summary.csv', index=False)