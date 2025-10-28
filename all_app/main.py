import sys
import os

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import FastAPI
from all_app.image import cifar100, fashion_mnist, mnist, allclass
from all_app.audio import gtzan, speech, urban, cars
from all_app.text import news, cods
import streamlit as st

app = FastAPI(title='ALL-Project')

app.include_router(cifar100.cifar_router)
app.include_router(fashion_mnist.fashion_router)
app.include_router(mnist.mnist_router)
app.include_router(allclass.allclass_router)
app.include_router(speech.speech_router)
app.include_router(cars.car_router)
app.include_router(gtzan.gtzan_router)
app.include_router(urban.urban_router)
app.include_router(news.news_router)
app.include_router(cods.code_router)


st.title('*')

with st.sidebar:
    st.header('Меню')
    name = st.radio('Задания', ['🎧Gtzan', '🎤Speech', '🚗Car', '🏙️Urban', '💯Cifar', '🌃Image', '👗Fashion',
                                '🎰Mnist', '📃News', '💻Code'])


if name == '🎧Gtzan':
    gtzan.gtzan_stream()

elif name == '🎤Speech':
    speech.speech_streamlit()

elif name == '🚗Car':
    cars.car_stream()

elif name == '🏙️Urban':
    urban.urban_stream()

elif name == '💯Cifar':
    cifar100.cifar_streamlit()

elif name == '🌃Image':
    allclass.allclass_streamlit()

elif name == '👗Fashion':
    fashion_mnist.fashion_streamlit()

elif name == '🎰Mnist':
    mnist.mnist_streamlit()

elif name == '📃News':
    news.news_streamlit()

elif name == '💻Code':
    cods.code_streamlit()
