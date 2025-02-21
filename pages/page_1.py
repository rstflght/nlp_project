# pages/page_01.py
import streamlit as st
import time
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score
from PIL import Image


st.title("Классификация отзывов на фильмы")

logo = Image.open('images/11.jpeg')
st.image(logo, width=800)

st.markdown("""
На этой странице вы можете ввести отзыв на фильм, и модели классифицируют его на одну из категорий: 
- **Good** (хороший), 
- **Neutral** (нейтральный), 
- **Bad** (плохой).
""")

user_input = st.text_area("Введите ваш отзыв на фильм:", "")

@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("models/face_yolo11m.pt")
    model = BertForSequenceClassification.from_pretrained("models/face_yolo11m.pt", num_labels=3)
    return tokenizer, model

def predict_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).item()
    return preds

if st.button("Классифицировать отзыв"):
    if user_input.strip() == "":
        st.error("Пожалуйста, введите отзыв.")
    else:
        # Загрузка моделей (заглушки)
        tokenizer, bert_model = load_bert_model()

        # Предсказание с использованием BERT
        start_time = time.time()
        bert_pred = predict_bert(user_input, tokenizer, bert_model)
        bert_time = time.time() - start_time

        # Заглушки для других моделей
        tfidf_pred = np.random.choice(["Good", "Neutral", "Bad"])
        tfidf_time = np.random.uniform(0.1, 0.5)

        lstm_pred = np.random.choice(["Good", "Neutral", "Bad"])
        lstm_time = np.random.uniform(0.2, 0.6)

        # Вывод результатов
        st.markdown("### Результаты предсказаний:")
        results = {
            "Модель": ["TF-IDF", "LSTM", "DistilBERT"],
            "Предсказание": [tfidf_pred, lstm_pred, ["Good", "Neutral", "Bad"][bert_pred]],
            "Время (сек)": [f"{tfidf_time:.4f}", f"{lstm_time:.4f}", f"{bert_time:.4f}"]
        }
        st.table(pd.DataFrame(results))

        # Сравнительная таблица по метрике F1-macro
        st.markdown("### Сравнение моделей по метрике F1-macro:")
        f1_scores = {
            "Модель": ["TF-IDF", "LSTM", "DistilBERT"],
            "F1-macro": [0.83, 0.88, 0.90]  # Заглушка
        }
        st.table(pd.DataFrame(f1_scores))

        # Диаграмма оценок внимания (заглушка)
        st.markdown("### Диаграмма оценок внимания для LSTM:")
        attention_weights = np.random.rand(10)  # Заглушка
        st.bar_chart(attention_weights)
