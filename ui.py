import streamlit as st


def show_data_button(data, button_label):
    if st.button(button_label):
        st.write(button_label)
        st.dataframe(data)
