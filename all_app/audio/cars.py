import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
import soundfile as sf
import streamlit as st
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
import io
from all_app.db.models import Car
from all_app.db.database import SessionLocal


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


car_router = APIRouter(prefix='/cars', tags=['cars'])

class CarsAudio(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4))
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 4 * 4, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 10)
    )
  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.MelSpectrogram(sample_rate=22050, n_mels=64)
max_len = 500
labels = torch.load('cars_labels3.pth')
index_to_label = {ind: lab for ind, lab in enumerate(labels)}
model = CarsAudio()
model.load_state_dict(torch.load('cars_model3.pth', map_location=device))
model.to(device)
model.eval()


def change_audio(waveform, sr):
    if sr != 22050:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(waveform)
    spec = transform(waveform)
    if spec.ndim == 3:
        spec = spec.mean(dim=0)
    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    elif spec.shape[1] < max_len:
        pad_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, pad_len))
    return spec


def car_stream():
    st.title('🚗 Распознавание звука автомобиля')
    st.text('Загрузите .wav файл или запишите звук двигателя, чтобы определить марку автомобиля')

    # === Варианты получения аудио ===
    file5 = st.file_uploader("Выберите аудиофайл (.wav)", type=['wav'], key="car_file")
    st.markdown("Или запишите звук прямо сейчас 🎙")
    audio_input = st.audio_input("🎤 Нажмите, чтобы записать звук", key="car_audio_input")

    audio_data = None
    audio_name = None

    if file5:
        audio_data = file5.read()
        audio_name = file5.name
        st.audio(audio_data, format="audio/wav")
    elif audio_input:
        audio_data = audio_input.getvalue()
        audio_name = "recorded_audio.wav"
        st.audio(audio_data, format="audio/wav")
    else:
        st.info("⚠️ Загрузите файл или запишите звук, чтобы продолжить.")
        return

    if st.button('🔍 Распознать звук'):
        try:
            if not audio_data:
                raise HTTPException(status_code=400, detail='Пустой аудиофайл')

            waveform, sr = sf.read(io.BytesIO(audio_data), dtype='float32')
            waveform = torch.tensor(waveform, dtype=torch.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)

            waveform = waveform.unsqueeze(0)
            spec = change_audio(waveform, sr).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                y_pred = model(spec)
                pred_ind = torch.argmax(y_pred, dim=1).item()
                pred_class = index_to_label[pred_ind]

            db: Session = next(get_db())
            car_db = Car(audio=audio_name, label=pred_class)
            db.add(car_db)
            db.commit()

            st.success(f'✅ Распознанный класс: **{pred_class}** (индекс: {pred_ind})')

        except Exception as e:
            st.error(f'❌ Ошибка: {e}')

