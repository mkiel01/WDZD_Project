import streamlit as st

def rename_columns_ui(data):
    new_column_names = {}
    for col in data.columns:
        new_name = st.text_input(f"Rename {col}", col)
        new_column_names[col] = new_name
    return new_column_names

def remove_columns_ui(data):
    columns_to_remove = st.multiselect("Select columns to remove", options=data.columns)
    return columns_to_remove

def add_column_ui():
    new_col_name = st.text_input("New column name")
    new_col_values = st.text_area("Enter values for the new column, separated by commas")
    return new_col_name, new_col_values

def adjust_data_size_ui():
    data_size_percentage = st.slider("Select the percentage of data to use", min_value=1, max_value=100, value=100)
    return data_size_percentage
