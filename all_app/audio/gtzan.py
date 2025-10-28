import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
import soundfile as sf
import io
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from all_app.db.models import Gtzan
from all_app.db.database import SessionLocal

# -----------------------------
# === Настройки и модель ===
# -----------------------------

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
transform = transforms.MelSpectrogram(sample_rate=22050, n_mels=64)
max_len = 500

genres = torch.load('labels_gtzan.pth')
index_to_label = {ind: lab for ind, lab in enumerate(genres)}

model = CheckAudio()
model.load_state_dict(torch.load('model_gtzan.pth', map_location=device))
model.to(device)
model.eval()


# -----------------------------
# === Функции обработки ===
# -----------------------------

def change_audio(waveform, sr):
    if sr != 22050:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(waveform)

    spec = transform(waveform).squeeze(0)
    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    if spec.shape[1] < max_len:
        pad_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, pad_len))
    return spec


# -----------------------------
# === Streamlit интерфейс ===
# -----------------------------

def gtzan_stream():
    """Современный UI для классификации жанра"""
    st.set_page_config(page_title="🎵 GTZAN Genre Classifier", layout="centered")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class^="css"] {
        background: radial-gradient(circle at top,#0e0f24 0%,#050610 70%) !important;
        font-family: 'Inter', sans-serif;
        color: #E4E4E7 !important;
    }

    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #F4F4F5;
        text-align: center;
    }

    .stButton>button {
        background: linear-gradient(90deg,#8b5cf6,#a78bfa);
        color: white;
        border: none;
        padding: 14px 34px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 0 18px rgba(139,92,246,0.45);
    }

    .stButton>button:hover {
        transform: scale(1.07);
        box-shadow: 0 0 30px rgba(167,139,250,0.85);
    }

    audio {
        border-radius: 10px;
        box-shadow: 0 0 12px rgba(124,58,237,0.4);
    }

    .footer {
        text-align:center;
        margin-top:40px;
        color:#a78bfa;
        font-size:14px;
    }

    .desc {
        text-align:center;
        font-size:16px;
        color:#C7C7C7;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2>🎶 Классификация музыкальных жанров</h2>", unsafe_allow_html=True)
    st.markdown("<p class='desc'>Загрузите файл или запишите аудио для определения жанра музыки</p>", unsafe_allow_html=True)

    # === Переключатель режимов ===
    if "gtzan_mode" not in st.session_state:
        st.session_state["gtzan_mode"] = "upload"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 Загрузить файл", key="upload_btn_gtzan", use_container_width=True):
            st.session_state["gtzan_mode"] = "upload"
    with col2:
        if st.button("🎤 Записать аудио", key="record_btn_gtzan", use_container_width=True):
            st.session_state["gtzan_mode"] = "record"

    mode = st.session_state["gtzan_mode"]

    st.markdown(
        f"<p style='text-align:center;font-size:15px;color:#b3b3b3;'>Текущий режим: <b>{'Загрузка файла' if mode == 'upload' else 'Запись с микрофона'}</b></p>",
        unsafe_allow_html=True)

    uploaded_file = None
    recorded_bytes = None

    if mode == "upload":
        uploaded_file = st.file_uploader("Выберите аудиофайл (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])
        if uploaded_file:
            st.audio(uploaded_file)
    else:
        recorded_bytes = audio_recorder()
        if recorded_bytes:
            st.audio(recorded_bytes, format="audio/wav")

    st.write("")

    if st.button("🚀 Распознать жанр"):
        try:
            data_bytes = uploaded_file.read() if mode == "upload" else recorded_bytes
            if not data_bytes:
                st.warning("⚠️ Пожалуйста, запишите или загрузите аудио перед анализом.")
                return

            waveform_np, sr = sf.read(io.BytesIO(data_bytes), dtype='float32')
            waveform = torch.tensor(waveform_np, dtype=torch.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)
            waveform = waveform.unsqueeze(0)

            spec = change_audio(waveform, sr).unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = model(spec)
                pred_ind = torch.argmax(y_pred, dim=1).item()
                pred_class = index_to_label[pred_ind]

            db: Session = next(get_db())
            new_gtzan = Gtzan(audio="stream_record.wav", label=pred_class)
            db.add(new_gtzan)
            db.commit()

            st.success(f"✅ Определён жанр: **{pred_class}** 🎧")

        except Exception as e:
            st.error(f"❌ Ошибка при анализе: {e}")

    st.markdown("<div class='footer'>Создано с 💜 для GTZAN Classifier</div>", unsafe_allow_html=True)
