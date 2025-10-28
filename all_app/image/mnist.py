from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from sqlalchemy.orm import Session
from fastapi import Depends
from all_app.db.models import Mnist
import streamlit as st

mnist_router = APIRouter(prefix='/mnist', tags=['mnist'])


from all_app.db.database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load("model_test.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


def mnist_streamlit():
    st.title('MNIST Classifier')
    file_image3 = st.file_uploader('Загрузите изображение', type=['png', 'jpeg', 'jpg', 'svg'])
    if not file_image3:
        st.error('error file')
    else:
        st.image(file_image3, caption='Загруженное изображение')
        if st.button('Определит Класс'):
            try:

                image_bytes = file_image3.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("L")
                image = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image)
                    predicted = torch.argmax(outputs, dim=1).item()

                db: Session = next(get_db())
                mnist_db = Mnist(
                    image=file_image3.name,
                    label=str(predicted)
                )
                db.add(mnist_db)
                db.commit()
                db.refresh(mnist_db)

                st.success({"predicted_class": predicted})

            except Exception as e:
                st.exception(f'Error: {e}')


