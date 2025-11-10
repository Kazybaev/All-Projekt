from fastapi import FastAPI, File, UploadFile, APIRouter, Depends
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from sqlalchemy.orm import Session
from all_app.db.models import Fashion
from all_app.db.database import SessionLocal
import streamlit as st


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


fashion_router = APIRouter(prefix='/fashion')

# class CheckImage(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.covn = nn.Sequential(
#         nn.Conv2d(1, 32, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Dropout(0.25),
#         nn.Conv2d(32, 64, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Dropout(0.25),
#
#     )
#     self.fc = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(64 * 7 * 7, 128),
#         nn.ReLU(),
#         nn.Dropout(0.25),
#         nn.Linear(128, 10),
#     )
#   def forward(self, x):
#     x = self.covn(x)
#     x = self.fc(x)
#     return x


class CheckImage(nn.Module):
  def __init__(self):
    super().__init__()
    self.covn = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),


    )
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
  def forward(self, x):
    x = self.covn(x)
    x = self.fc(x)
    return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CheckImage()
model.load_state_dict(torch.load("model_v4.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


image_dd = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
# uvicorn main:app --reload

def fashion_streamlit():
    st.title('Fashion Mnist Classifier')
    file_image2 = st.file_uploader('Загрузите изображение', type=['png', 'jpeg', 'jpg', 'svg'])

    if not file_image2:
        st.error('Загрузите файл!')
    else:
        st.image(file_image2, caption='Загруженное изображение')

        if st.button('Определить класс'):
            try:
                # Read image bytes
                image_bytes = file_image2.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("L")
                image = transform(image).unsqueeze(0).to(device)

                # Prediction
                with torch.no_grad():
                    outputs = model(image)
                    predicted = torch.argmax(outputs, dim=1).item()
                    label = image_dd[predicted]

                # Save to DB
                db: Session = next(get_db())

                record = Fashion(
                    image=file_image2.name,
                    label=label
                )
                db.add(record)
                db.commit()
                db.refresh(record)

                st.success({
                    "id": record.id,
                    "class": label,
                    "index": predicted
                })


                # if predicted == 0:
                #     return {"T-Shirt": predicted}
                # elif predicted == 1:
                #     return {"Trouser": predicted}
                # elif predicted == 2:
                #     return {"Pullover": predicted}
                # elif predicted == 3:
                #     return {"Dress": predicted}
                # elif predicted == 4:
                #     return {"Coat": predicted}
                # elif predicted == 5:
                #     return {"Sandal": predicted}
                # elif predicted == 6:
                #     return {"Shirt": predicted}
                # elif predicted == 7:
                #     return {"Sneaker": predicted}
                # elif predicted == 8:
                #     return {"Bag": predicted}
                # elif predicted == 9:
                #     return {"Ankle boot": predicted}

            except Exception as e:
                st.exception(f'Error: {e}')
