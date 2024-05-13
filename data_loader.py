import pandas as pd
import streamlit as st

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

def save_column_changes(data, new_column_names):
    data.rename(columns=new_column_names, inplace=True)
    return data

def remove_columns(data, columns_to_remove):
    data.drop(columns=columns_to_remove, inplace=True)
    return data

def add_column(data, new_col_name, new_col_values):
    new_values_list = [val.strip() for val in new_col_values.split(",")]
    if len(new_values_list) == len(data):
        data[new_col_name] = new_values_list
    else:
        st.error("The number of values does not match the number of rows in the dataset")
    return data
