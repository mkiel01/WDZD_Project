import streamlit as st


def adjust_data_size_ui(data_len):
    desired_data_size = st.slider(
        "Adjust desired data size",
        min_value=1,
        max_value=data_len,
        value=data_len,
    )
    return desired_data_size


def show_data_button(data, button_label):
    if st.button(button_label):
        st.write(button_label)
        st.dataframe(data)
