import torch
import torch.nn as nn
from pydantic import BaseModel
from fastapi import APIRouter
from sqlalchemy.orm import Session
import torchaudio
import torchaudio.transforms as transforms
import soundfile as sf
import io
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from all_app.db.models import Speech
from all_app.db.database import SessionLocal

# === Настройки и модель ===

labels = torch.load("labels.pth")
num_classes = len(labels)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


speech_router = APIRouter(prefix="/speech", tags=["speech"])


class CheckAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.second(self.first(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CheckAudio()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.MelSpectrogram(sample_rate=16000, n_mels=32)


# ---------------------------
# === Вспомогательные функции ===
# ---------------------------

def process_audio_bytes(data_bytes: bytes, target_sr: int = 16000):
    """Читает аудио из байтов и приводит к нужному sample rate"""
    audio_np, sr = sf.read(io.BytesIO(data_bytes), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def predict_from_waveform(waveform: torch.Tensor):
    spec = transform(waveform)
    spec = spec.unsqueeze(1).to(device)
    with torch.no_grad():
        outputs = model(spec)
        predicted = torch.argmax(outputs, dim=1).item()
        return labels[predicted]


# ---------------------------
# === Streamlit UI ===
# ---------------------------

def speech_streamlit():
    """Красивая современная страница Streamlit"""
    st.set_page_config(page_title="🎧 Speech Recognizer", layout="centered")

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

    .button-row {
        display: flex;
        justify-content: center;
        gap: 25px;
        margin: 30px 0 40px 0;
        flex-wrap: wrap;
    }

    .mode-btn {
        background: linear-gradient(90deg,#7c3aed,#a78bfa);
        border: none;
        color: white;
        padding: 15px 36px;
        font-size: 17px;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.35s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 15px rgba(124,58,237,0.35);
    }

    .mode-btn::before {
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(120deg, rgba(255,255,255,0.25), transparent, rgba(255,255,255,0.15));
        transition: all 0.4s ease;
    }

    .mode-btn:hover::before { left: 100%; }
    .mode-btn:hover {
        transform: scale(1.07);
        box-shadow: 0 0 25px rgba(167,139,250,0.8);
    }

    .active-btn {
        background: linear-gradient(90deg,#a78bfa,#c084fc);
        box-shadow: 0 0 25px rgba(167,139,250,0.9);
        transform: scale(1.05);
    }

    .stButton>button {
        background: linear-gradient(90deg,#8b5cf6,#a78bfa);
        color: white;
        border: 0;
        padding: 14px 32px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 16px;
        transition: 0.3s ease;
        box-shadow: 0 0 18px rgba(139,92,246,0.45);
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 28px rgba(167,139,250,0.85);
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
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2>🎙️ Распознавание голосовых команд</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;font-size:16px;color:#C7C7C7;'>Выберите режим: загрузка файла или запись голоса</p>",
        unsafe_allow_html=True)

    # === Переключатель режимов ===
    if "mode" not in st.session_state:
        st.session_state["mode"] = "upload"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 Загрузить файл", key="upload_btn", use_container_width=True):
            st.session_state["mode"] = "upload"
    with col2:
        if st.button("🎤 Записать голос", key="record_btn", use_container_width=True):
            st.session_state["mode"] = "record"

    mode = st.session_state["mode"]

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

    st.write("")  # небольшой отступ

    if st.button("🚀 Распознать голос"):
        try:
            data_bytes = uploaded_file.read() if mode == "upload" else recorded_bytes
            if not data_bytes:
                st.warning("⚠️ Пожалуйста, запишите или загрузите аудио перед распознаванием.")
                return

            waveform, sr = process_audio_bytes(data_bytes)
            label = predict_from_waveform(waveform)

            db: Session = next(get_db())
            new_speech = Speech(audio="stream_record.wav", label=label)
            db.add(new_speech)
            db.commit()

            st.success(f"✅ Распознанная команда: **{label}**")
        except Exception as e:
            st.error(f"❌ Ошибка при распознавании: {e}")

    st.markdown("<div class='footer'>Создано с ❤️ для проекта Speech Recognizer</div>", unsafe_allow_html=True)
