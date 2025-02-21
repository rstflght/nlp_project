# pages/page_03.py
import streamlit as st
from PIL import Image


# Заголовок страницы
st.title("Генерация текста GPT")

logo = Image.open('images/лого_3.jpeg')
st.image(logo, width=800)

st.markdown("""
На этой странице вы можете сгенерировать текст с помощью GPT-модели.
""")

