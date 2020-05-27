import pandas as pd

def data_is_available(file_name):
    dataset = pd.read_csv(file_name)
    if dataset.empty:
        return False
    else:
        return dataset.shape
