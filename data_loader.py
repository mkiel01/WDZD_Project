import pandas as pd


def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data
