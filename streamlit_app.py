import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Загрузка данных из CSV файла
@st.cache
def load_data(filename):
    df = pd.read_csv(filename)
    return df

# Загрузка модели BERT и токенизатора
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(model_name_or_path):
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(model_name_or_path)
    return tokenizer, model

# Словарь для преобразования меток в категории на русском языке
label_to_category = {
    0: 'Здравоохранение',
    1: 'ЖКХ',
    2: 'Образование',
    3: 'Инфраструктура',
    4: 'Культура',
    5: 'Экология',
    6: 'Социальное обеспечение',
    7: 'Политика',
    8: 'Безопасность',
    9: 'Доступность товаров и услуг',
    10: 'Официальные заявления',
    11: 'Туризм',
    12: 'Факты'
}

# Функция для предсказания категории текста
def predict_category(text, tokenizer, model):
    # Токенизируем текст
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Получаем предсказания от модели
    with torch.no_grad():
        outputs = model(**inputs)

    # Получаем предсказанную категорию
    predicted_label = torch.argmax(outputs.logits).item()
    
    # Преобразование метки в категорию на русском языке
    predicted_category = label_to_category[predicted_label]
    return predicted_category

# Главный код Streamlit
def main():
    st.set_page_config(layout="wide")

    st.title('Веб-сервис анализа данных и текстов с BERT')

    # Опции для выбора вкладок
    options = ['Аналитика данных', 'Проверка модели']
    choice = st.sidebar.selectbox('Выберите вкладку', options)

    # Отображение соответствующей вкладки
    if choice == 'Аналитика данных':
        # Загрузка данных из CSV файла
        filename = st.file_uploader("Загрузите ваш файл CSV", type=['csv'])
        if filename is not None:
            df = load_data(filename)
            # Выводим датасет
            st.subheader('Датасет:')
            st.write(df)
            
            # Выводим статистику по датасету
            st.subheader('Статистика по датасету:')
            st.write(df.describe())
    elif choice == 'Проверка модели':
        st.subheader('Проверка модели BERT:')
        # Загрузка модели и токенизатора
        model_name_or_path = "/app/model" # Укажите путь к вашей модели
        tokenizer, model = load_model_and_tokenizer(model_name_or_path)

        # Ввод текста пользователем
        text_input = st.text_area('Введите текст для анализа')

        # Предсказание категории при нажатии на кнопку
        if st.button('Анализировать'):
            if text_input:
                # Предсказываем категорию текста
                predicted_category = predict_category(text_input, tokenizer, model)
                st.write('Предсказанная категория текста:', predicted_category)
            else:
                st.write('Пожалуйста, введите текст для анализа')

if __name__ == '__main__':
    main()
