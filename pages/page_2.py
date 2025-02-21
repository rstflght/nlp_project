# pages/page_02.py
import streamlit as st
from PIL import Image


# Заголовок страницы
st.title("Оценка степени токсичности сообщений")

logo = Image.open('images/2.jpeg')
st.image(logo, width=800)

st.markdown("""
На этой странице вы можете оценить степень токсичности вашего сообщения с помощью модели Руберта-Тини-токсичности.
""")


