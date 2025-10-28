from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter, Depends
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import io
import soundfile as sf
from all_app.db.database import SessionLocal
from all_app.db.models import Urban
from sqlalchemy.orm import Session
import streamlit as st


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Urban_Music(nn.Module):
    def __init__(self, num_classes=2):  # по умолчанию 2 класса для заглушки
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((64, 64)),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((32, 32)),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*32*32, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.second(x)
        return x


BASE_DIR = Path(__file__).resolve().parent.parent.parent
label_urban = BASE_DIR / 'label_urban.pth'
model_urban = BASE_DIR / 'model_urban.pth'


if not label_urban.exists() or not model_urban.exists():
    print(f"[INFO] Файлы не найдены. Создаем заглушки: {label_urban}, {model_urban}")
    torch.save(['class1', 'class2'], label_urban)
    dummy_model = Urban_Music(num_classes=2)
    torch.save(dummy_model.state_dict(), model_urban)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=64)
max_len = 500


classes = torch.load(label_urban)
index_to_label = {ind: lab for ind, lab in enumerate(classes)}
model = Urban_Music(num_classes=len(classes))
model.load_state_dict(torch.load(model_urban, map_location=device))
model.to(device)
model.eval()


def change_audio(waveform, sr):
    if sr != 22050:
        waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
    spec = transform(waveform)
    if spec.shape[-1] > max_len:
        spec = spec[:, :, :max_len]
    elif spec.shape[-1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[-1]))
    return spec.squeeze(0)


urban_router = APIRouter(prefix="/urban", tags=["Urban"])


def urban_stream():
    st.title('Urban Speech Audio')
    st.text('Загрузите аудиофайл (.wav) для расспознования ')

    file3 = st.file_uploader("Выберите аудиофайл", type=['wav'], key="urban_file")

    if not file3:
        st.info('Загрузите аудиофайл')
    else:
        st.audio(file3)
        if st.button('Распознать'):

            try:
                data = file3.read()
                if not data:
                    raise HTTPException(status_code=400, detail='Пустой файл')
                wf, sr = sf.read(io.BytesIO(data), dtype='float32')
                if wf.ndim == 2:
                    wf = wf.mean(axis=1)
                wf = torch.tensor(wf).unsqueeze(0)
                spec = change_audio(wf, sr).unsqueeze(0).to(device)
                with torch.no_grad():
                    y_pred = model(spec)
                    pred_ind = torch.argmax(y_pred, dim=1).item()
                    pred_class = index_to_label[pred_ind]

                db: Session = next(get_db())
                urban_db = Urban(audio=file3.name,
                                 label=pred_class)
                db.add(urban_db)
                db.commit()
                db.refresh(urban_db)

                st.success(f'answer_class:  {pred_class}')

            except Exception as e:
                st.exception(f'Error: {e}')
