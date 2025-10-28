from fastapi import FastAPI, APIRouter, Depends
import torch
import torch.nn as nn
from pydantic import BaseModel
from torchtext.data import get_tokenizer
from googletrans import Translator
from sqlalchemy.orm import Session

from all_app.db.database import SessionLocal
from all_app.db.models import News
import streamlit as st

news_router = APIRouter(prefix='/news', tags=['news'])

# /////////////////////
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class_name = {
    0: 'World',
    1: 'Sports',
    2: 'Business',
    3: 'Sci/Tech'
}


class CheckNews(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.lin = nn.Linear(128, 4)

    def forward(self, x):
        x = self.emb(x)
        _, (x, _) = self.lstm(x)
        x = x[-1]
        x = self.lin(x)
        return x


vocab = torch.load('vocab_news.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckNews(len(vocab))
model.load_state_dict(torch.load('model_news.pth', map_location=device))
model.to(device)
model.eval()

translator = Translator()
tokenizer = get_tokenizer('basic_english')


def change_word(text):
    return [vocab[i] for i in tokenizer(text)]


class TextSchema(BaseModel):
    text: str

def news_streamlit():
    st.title('💌 AG News Text Classifier')

    input_text = st.text_area('Введите текст здесь :')

    if st.button('Классифицировать'):
            try:
                translated = translator.translate(input_text, dest='en')
                translated_text = translated.text

                num_text = torch.tensor(change_word(translated_text)).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(num_text)
                    result = torch.argmax(pred, dim=1).item()
                    predicted_class = class_name[result]

                db: Session = next(get_db())
                news_db = News(
                    text=input_text,
                    translated_text=translated_text,
                    label=predicted_class
                )
                db.add(news_db)
                db.commit()
                db.refresh(news_db)

                st.success(f"✅ Класс: {predicted_class}")

            except Exception as e:
                st.exception(f'Error: {e}')
