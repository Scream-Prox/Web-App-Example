import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from datetime import datetime
import base64
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Установка конфигурации страницы в самом начале скрипта
st.set_page_config(layout="wide")

def image_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode() # Кодируем изображение в base64 и декодируем в строку
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center center;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Установка фонового изображения
image_local('static/image/fon.png') # Указываем путь к вашему изображению

def format_date_ru(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    month_names = {
        1: 'Января', 2: 'Февраля', 3: 'Марта', 4: 'Апреля', 5: 'Мая', 6: 'Июня',
        7: 'Июля', 8: 'Августа', 9: 'Сентября', 10: 'Октября', 11: 'Ноября', 12: 'Декабря'
    }
    formatted_date_time = date_obj.strftime(f"%d {month_names[date_obj.month]} %H:%M")
    return formatted_date_time

@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename)
    return df

@st.cache_resource
def load_model_and_tokenizer(model_name_or_path):
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(model_name_or_path)
    return tokenizer, model

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

def predict_category(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():

        outputs = model(**inputs)

    predicted_label = torch.argmax(outputs.logits).item()

    predicted_category = label_to_category[predicted_label]

    return predicted_category

def calculate_average_tonality(df):
    # Создаем словарь для преобразования строковых значений тональности в числовые
    tonality_mapping = {'Positive': 7, 'Negative': 3, 'Neutral': 5}
    
    # Применяем словарь к столбцу 'Tonality'
    df['Tonality'] = df['Tonality'].map(tonality_mapping)
    
    # Вычисляем среднее значение для каждой категории
    average_tonality = df.groupby('Category')['Tonality'].mean()
    
    return average_tonality


def draw_donut_chart(score, total_score=10, edgecolor='none', category='Общий рейтинг', diagram_title='', title_color='white', score_color='white', category_color='white'):
    
    fraction = score / total_score

    if fraction < 0 or fraction > 1:
        fraction = 0

    

    fig, ax = plt.subplots(facecolor='none')

    ax.axis('off')
    sector_color = (2/255, 67/255, 157/255, 1)
    wedges, texts = ax.pie([fraction, 1-fraction], colors=[sector_color, 'none'], startangle=90)
    plt.setp(wedges, width=0.2, edgecolor=edgecolor)

    ax.text(0, -0.2, diagram_title, fontsize=12, ha='center', va='top', color=title_color)

    ax.text(0, 0.2, category, fontsize=12, ha='center', va='bottom', color=category_color)

    ax.text(0, 0, f'{score:.1f}', fontsize=20, ha='center', va='center', color=score_color)
    return fig

def main():
    st.title('Веб-сервис анализа данных и текстов с BERT')
    options = ['Аналитика данных', 'Проверка модели']
    choice = st.sidebar.selectbox('Выберите вкладку', options, key='main_choice')

    if choice == 'Аналитика данных':
        filename = 'final_verse.csv'
        df = load_data(filename)
        st.markdown('Датасет:')
        st.write(df)
        st.subheader('Статистика по датасету:')
        st.write(df.describe())
        cols = st.columns(1)
        with cols[0]:
            st.header('Самые просматриваемые посты')
            # Добавляем уникальный ключ для виджета выбора категории
            category = st.selectbox('Выберите категорию', df['Category'].unique(), key='top_posts_category')
            filtered_df = df[df['Category'] == category]
            top_posts = filtered_df.sort_values(by='Views', ascending=False).head(5)
            for index, row in top_posts.iterrows():
                post_text = row['Text']
                post_date = row['Data'] + ' ' + row['Time']
                formatted_date = format_date_ru(post_date)
                st.markdown(f'<div style="padding: 10px; background-color: rgba(4,20,61, 0.5); margin-bottom: 10px; border-radius: 10px; box-sizing: border-box; color: white;">'
                            f'<span style="color: #A0AEC0;">{formatted_date}</span><br>'
                            f'{post_text}</div>', unsafe_allow_html=True)
        
        average_tonality = calculate_average_tonality(df)
        # Добавляем уникальный ключ для виджета выбора категории в круговой диаграмме
        
        category = st.selectbox('Выберите категорию', df['Category'].unique(), key='donut_chart_category')
        score = average_tonality[category]

        
        fig = draw_donut_chart(score, edgecolor='none', category=category, diagram_title='Общий рейтинг', title_color='white', score_color='white', category_color='white')
        st.pyplot(fig)


    elif choice == 'Проверка модели':
        st.subheader('Проверка модели BERT:')
        model_name_or_path = "/app/model"
        tokenizer, model = load_model_and_tokenizer(model_name_or_path)
        text_input = st.text_area('Введите текст для анализа', key='model_check_text_input')
        if st.button('Анализировать', key='model_check_analyze_button'):
            if text_input:
                predicted_category = predict_category(text_input, tokenizer, model)
                st.write('Предсказанная категория текста:', predicted_category)
            else:
                st.write('Пожалуйста, введите текст для анализа')

if __name__ == '__main__':
    main()
