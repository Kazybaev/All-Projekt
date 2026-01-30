from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from sqlalchemy.orm import Session
from fastapi import Depends
from all_app.db.models import AllClass
import streamlit as st

allclass_router = APIRouter(prefix='/all_class', tags=['all_class'])


from all_app.db.database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class CheckImageAlexNET(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 56 * 56, 128),
        nn.ReLU(),
        nn.Linear(128, 19)
    )

  def forward(self, x):
    first_x = self.first(x)
    second_x = self.second(first_x)
    return second_x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CheckImageAlexNET()
model.load_state_dict(torch.load("my_model.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_dd = [
    'Pebbles',
    'Shells',
    'airplane',
    'bear',
    'bike',
    'car',
    'cat',
    'dog',
    'elephant',
    'helicopter',
    'horse',
    'laptop',
    'lion',
    'lower_clothes',
    'panda',
    'phone',
    'scooter',
    'ship',
    'upper_clothes'
]


def allclass_streamlit():
    st.title('All Class Classifier')
    file_image4 = st.file_uploader('Загрузите изображение', type=['png', 'jpeg', 'jpg', 'svg'])
    if not file_image4:
        st.error('error file')
    else:
        st.image(file_image4, caption='Загруженное изображение')
        if st.button('Определит Класс'):
            try:

                image_bytes = file_image4.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image)
                    predicted = torch.argmax(outputs, dim=1).item()
                    label = image_dd[predicted]

                db: Session = next(get_db())
                mnist_db = AllClass(
                    image=file_image4.name,
                    label=label
                )
                db.add(mnist_db)
                db.commit()
                db.refresh(mnist_db)

                st.success(f"predicted_class: {label}")

            except Exception as e:
                st.exception(f'Error: {e}')


