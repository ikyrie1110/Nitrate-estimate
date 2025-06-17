import pandas as pd

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data):
    data = data.drop(columns=['station'])
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.fillna(data.mean())  
    data['Year'] = data['DATE'].astype(str).str[:4].astype(int)
    data['Month'] = data['DATE'].astype(str).str[-2:].astype(int)
    data['Quarter'] = (data['Month'] - 1) // 3 + 1
    data['Year_Quarter'] = data['Year'].astype(str) + "_Q" + data['Quarter'].astype(str)
    return data
