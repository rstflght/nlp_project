import streamlit as st
from PIL import Image

st.title("Команда LSTM")
st.subheader("Обработка естественного языка")

# Загрузка логотипа
logo = Image.open('images/лого_0.jpeg')
st.image(logo, width=600)

st.markdown("""
### Оглавление:
1. [Классификация отзывов на фильмы](#страница-1)
    - Датасет содержит отзывы о фильмах.
    - Модели: TF-IDF, LSTM, DistilBERT.
2. [Оценка степени токсичности сообщений](#страница-2)
    - Используется модель Руберта-Тини-токсичности.
3. [Генерация текста GPT](#страница-3)
    - Пользователь может контролировать генерацию текста.
""")

st.sidebar.title("Команда проекта:")
st.sidebar.markdown("[Сергей](https://github.com/rstflght)")
st.sidebar.markdown("[Маша](https://github.com/evcranberry)")
st.sidebar.markdown("[Даша](https://github.com/DashonokOk)")


  git config --global user.email "akylovadashaa@gmail.com"
  git config --global user.name "DashonokOk"
