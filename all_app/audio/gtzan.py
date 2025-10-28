import torchaudio
from fastapi import HTTPException, APIRouter
import torch
import torch.nn as nn
from torchaudio import transforms
import torch.nn.functional as F
import io
import soundfile as sf

from all_app.db.models import Gtzan
from sqlalchemy.orm import Session
from all_app.db.database import SessionLocal
import streamlit as st
from audio_recorder_streamlit import audio_recorder

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


gtzan_router = APIRouter(prefix='/gtzan', tags=['gtzan'])


class CheckAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.second(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tranform = transforms.MelSpectrogram(sample_rate=22050, n_mels=64)
max_len = 500

genres = torch.load('labels_gtzan.pth')
index_to_label = {ind: lab for ind, lab in enumerate(genres)}

model = CheckAudio()
model.load_state_dict(torch.load('model_gtzan.pth', map_location=device))
model.to(device)
model.eval()


def change_audio(waveform, sr):
    if sr != 22050:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(waveform)

    spec = tranform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    if spec.shape[1] < max_len:
        pad_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, pad_len))

    return spec


def gtzan_stream():
    st.title('🎵 Gtzan Music Genre Classifier')
    st.text('Загрузите аудиофайл (.wav) для распознавания жанра')

    file2 = st.file_uploader("Выберите аудиофайл", type=['wav'], key="gtzan_file")
    if not file2:
        st.info('⚠️ Загрузите .wav файл')

    audio_bytes = audio_recorder()
    if not audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
    else:
        st.audio(audio_bytes)
        if st.button('Распознать жанр'):
            try:
                data = audio_bytes.read()
                if not data:
                    raise HTTPException(status_code=400, detail='Пустой файл')

                waveform, sr = sf.read(io.BytesIO(data), dtype='float32')
                waveform = torch.tensor(waveform, dtype=torch.float32)

                if waveform.ndim > 1:
                    waveform = waveform.mean(dim=1)

                waveform = waveform.unsqueeze(0)

                spec = change_audio(waveform, sr).unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred = model(spec)
                    pred_ind = torch.argmax(y_pred, dim=1).item()
                    pred_class = index_to_label[pred_ind]

                db: Session = next(get_db())
                gtzan_db = Gtzan(audio=file2.name, label=pred_class)
                db.add(gtzan_db)
                db.commit()

                st.success(f'✅ Жанр: {pred_class} (индекс: {pred_ind})')

            except Exception as e:
                st.error(f'Ошибка: {e}')
