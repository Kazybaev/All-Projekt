import io
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import APIRouter
from sqlalchemy.orm import Session
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from all_app.db.database import SessionLocal
from all_app.db.models import Cifar100

cifar_router = APIRouter(prefix='/cifar100', tags=['cifar'])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---- Your model definition (kept as in your project) ----
class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---- Model loading: adjust path if needed ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CheckImage()
model.load_state_dict(torch.load("model_pk.pth", map_location=device))
model.to(device)
model.eval()


# ---- CIFAR100 labels list (same as yours) ----
image_dd = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
    "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
    "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
    "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
    "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

# ---- Preprocessing pipeline: resize to 32x32 and normalize (grayscale->RGB kept)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


# ---- Helper: convert canvas data (base image) to PIL Image ----
def canvas_data_to_pil(image_data, invert=True):
    """image_data: numpy array HxWx4 (RGBA) from st_canvas
    We'll convert to RGB and invert colors optionally so white brush becomes foreground on dark bg.
    """
    if image_data is None:
        return None
    # Convert uint8 to Image
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('RGB')

    # If canvas background is dark and brush is white, model expects white-on-black or white-on-dark?
    # We will invert to make drawn white strokes on black background -> white strokes become white on black.
    if invert:
        # Convert to grayscale and invert only luminance to keep strokes
        img = ImageOps.invert(img)
    return img


# ---- Main Streamlit page ----

def cifar_streamlit():
    st.set_page_config(page_title='CIFAR100 — Classifier', layout='centered')

    # Reuse the same styles as in speech_streamlit
    st.markdown("""
    <style>
    :root{
      --bg:#050610; --card:#0d1220; --text:#e4e4e7;
      --accent1:#7c3aed; --accent2:#a78bfa; --accentGlow:#8b5cf6;
    }
    html, body, [class^="css"] {
        background: radial-gradient(circle at top,#0a0f1f 0%,#050610 60%) !important;
        color: var(--text) !important;
    }
    .main-block {
        max-width: 900px;
        margin: 0 auto;
        padding-top: 10px;
    }
    .block {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(124,58,237,0.2);
        border-radius: 14px;
        padding: 22px;
        margin-top: 20px;
        box-shadow: 0 0 15px rgba(124,58,237,0.25);
    }
    .stButton>button {
        background: linear-gradient(90deg,var(--accent1),var(--accent2));
        color: white;
        border: 0;
        padding: 10px 22px;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.25s;
        box-shadow: 0 0 15px rgba(124,58,237,0.4);
    }
    .stButton>button:hover {
        transform: scale(1.04);
        box-shadow: 0 0 25px rgba(124,58,237,0.8);
    }
    .mode-btn {
        width:100%;
        padding:10px 12px;
        border-radius:10px;
        background: linear-gradient(90deg,var(--accent1),var(--accent2));
        color:white; font-weight:600;
        box-shadow: 0 6px 18px rgba(124,58,237,0.16);
    }
    .result-card {
        border-radius: 12px;
        padding: 18px;
        background: linear-gradient(180deg, rgba(124,58,237,0.12), rgba(167,139,250,0.06));
        box-shadow: 0 8px 30px rgba(124,58,237,0.12);
        text-align:center;
    }
    .result-title { font-size:22px; font-weight:700; }
    .result-label { font-size:34px; font-weight:800; margin-top:6px; }
    .try-again { margin-top:12px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-block">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center;">🖼️ CIFAR-100 Classifier</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:var(--text);">Загрузите изображение или нарисуйте его на холсте</p>', unsafe_allow_html=True)

    st.markdown('<div class="block">', unsafe_allow_html=True)

    # Mode buttons (upload / draw)
    col1, col2 = st.columns([1, 1])
    with col1:
        if 'cifar_mode' not in st.session_state:
            st.session_state['cifar_mode'] = 'upload'
        if st.button('📁 Загрузить изображение', key='cifar_upload_btn'):
            st.session_state['cifar_mode'] = 'upload'
    with col2:
        if st.button('🎨 Нарисовать', key='cifar_draw_btn'):
            st.session_state['cifar_mode'] = 'draw'

    mode = st.session_state.get('cifar_mode', 'upload')

    uploaded_file = None
    drawn_image = None

    # Layout: left - input, right - preview + actions
    left_col, right_col = st.columns([1, 1])

    with left_col:
        if mode == 'upload':
            st.markdown('**Режим — загрузка файла**')
            uploaded_file = st.file_uploader('Загрузите изображение', type=['png', 'jpeg', 'jpg', 'svg'])
        else:
            st.markdown('**Режим — рисование**')

            # Canvas settings
            canvas_result = st_canvas(
                fill_color="#00000000",  # Transparent fill
                stroke_width=12,
                stroke_color="#FFFFFF",  # белая кисть по умолчанию
                background_color="#000000",
                width=256,
                height=256,
                drawing_mode="freedraw",
                key="canvas",
            )

            if canvas_result and canvas_result.image_data is not None:
                drawn_image = canvas_data_to_pil(canvas_result.image_data, invert=True)

    with right_col:
        st.markdown('**Превью**')
        if mode == 'upload' and uploaded_file:
            try:
                img_bytes = uploaded_file.read()
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                st.image(pil_img, caption='Загруженное изображение', use_column_width=True)
            except Exception as e:
                st.error(f'Не удалось открыть изображение: {e}')
        elif mode == 'draw' and drawn_image:
            st.image(drawn_image, caption='Нарисованное изображение (после инвертирования)', use_column_width=True)
        else:
            st.info('Здесь появится превью загруженного или нарисованного изображения')

    # Recognize button
    if st.button('🔮 Определить класс', key='cifar_recognize'):
        try:
            # Choose source
            if mode == 'upload':
                if not uploaded_file:
                    st.error('Нет загруженного файла')
                else:
                    img_bytes = uploaded_file.read()
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            else:
                if drawn_image is None:
                    st.error('Нет рисунка на холсте')
                    return
                pil_img = drawn_image

            # Preprocess: make grayscale->RGB (train_transform handles it) and resize to 32x32
            input_tensor = train_transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                predicted = torch.argmax(outputs, dim=1).item()
                label = image_dd[predicted]

            # Save to DB
            db: Session = next(get_db())
            image_name = uploaded_file.name if uploaded_file else 'canvas_draw.png'
            cifar_db = Cifar100(image=image_name, label=label)
            db.add(cifar_db)
            db.commit()
            db.refresh(cifar_db)

            # ВАУ-результат
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-title">✅ Результат распознавания</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-label">{label.upper()}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="margin-top:8px;">Класс #{predicted} • confidence unavailable</div>', unsafe_allow_html=True)
            st.markdown('<div class="try-again">', unsafe_allow_html=True)
            if st.button('🔁 Попробовать ещё', key='try_again'):
                # Just a soft reset of preview states
                st.session_state['cifar_mode'] = 'upload'
                st.experimental_rerun()
            st.markdown('</div></div>', unsafe_allow_html=True)

        except Exception as e:
            st.exception(f'Ошибка при распознавании: {e}')

    st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; margin-top:12px; color:#a78bfa">Made with ❤️ — CIFAR-100 Recognizer</div>', unsafe_allow_html=True)


# If run directly, show the page
if __name__ == '__main__':
    cifar_streamlit()
