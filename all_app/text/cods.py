from fastapi import APIRouter, Depends
import torch
import torch.nn as nn
from pydantic import BaseModel
from sqlalchemy.orm import Session
from torchtext.data import get_tokenizer
import streamlit as st

from all_app.db.database import SessionLocal
from all_app.db.models import Code


code_router = APIRouter(prefix="/cods", tags=["cods"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class_names = {
    "JavaScript",
    "swift",
    "C++",
    "Go"
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


vocab = torch.load("cod_vocabs.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CheckNews(len(vocab))
model.load_state_dict(torch.load("cod_models.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = get_tokenizer("basic_english")


def text_to_tensor(text):
    tokens = tokenizer(text)
    ids = [vocab[t] if t in vocab else 0 for t in tokens]
    return torch.tensor(ids).unsqueeze(0).to(device)


class CodeSchema(BaseModel):
    text: str


def code_streamlit():
    st.title("💻 Code Language Classifier")

    input_code = st.text_area("Вставь свой код сюда:")

    if st.button("Определить код"):
        try:
            input_tensor = text_to_tensor(input_code)

            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                predicted_class = class_names[pred]

            db: Session = next(get_db())
            code_sample = Code(
                text=input_code,
                label=predicted_class
            )
            db.add(code_sample)
            db.commit()
            db.refresh(code_sample)

            st.success(f"✅ Определён язык: {predicted_class}")

        except Exception as e:
            st.exception(f"❌ Ошибка: {e}")
