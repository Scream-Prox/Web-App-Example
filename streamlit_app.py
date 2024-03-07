import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

import datetime
from datetime import datetime


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
        filename = 'final_verse.csv'
        df = load_data(filename)
        # Выводим датасет
        st.markdown('Датасет:')
        st.write(df)
        
        # Выводим статистику по датасету
        st.subheader('Статистика по датасету:')
        st.write(df.describe())

        # Создаем колонку с шириной 100%
        cols = st.columns(1)

        with cols[0]:
            st.header('Самые просматриваемые посты')
    # Выпадающий список для выбора категории
            category = st.selectbox('Выберите категорию', df['Category'].unique())
    # Фильтрация датафрейма по выбранной категории
            filtered_df = df[df['Category'] == category]
    # Сортировка по количеству просмотров и выборка 5 самых популярных постов
            top_posts = filtered_df.sort_values(by='Views', ascending=False).head(5)
    
            for index, row in top_posts.iterrows():
                post_text = row['Text']
                post_date = row['Data'] + ' ' + row['Time']
        # Преобразование строки даты и времени в объект datetime
                post_datetime = datetime.strptime(post_date, '%Y-%m-%d %H:%M:%S')
        # Форматирование даты и времени
                formatted_date = post_datetime.strftime('%d %B %H:%M')
        
        # Создание строки с текстом поста и датой публикации
                post_with_date = f"{post_text}\n\nДата публикации: {formatted_date}"
        
        # Вывод текста поста с закругленной обводкой и датой публикации
                st.markdown(f'<div style="border: 1px solid white; padding: 10px; margin-bottom: 10px; border-radius: 10px; box-sizing: border-box;">{post_with_date}</div>', unsafe_allow_html=True)

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
