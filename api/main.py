

from fastapi import FastAPI, File, UploadFile # Fast API for making the server
from fastapi.middleware.cors import CORSMiddleware # cors making making connection between frontend port and backend port
import uvicorn # for running the file and start the server
import numpy as np 
from io import BytesIO # for open the image file in binary format
from PIL import Image
import tensorflow as tf

app = FastAPI() # initializeing the server

origins = [
    "http://localhost", # this allow every port
    "http://localhost:3000", # delcaring the port number for the frontend 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) # middleware sits before your API endpoints, handling requests and responses.

MODEL = tf.keras.models.load_model("../saved_models/1") # importing the model

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

RESTful API = REST = Representational State Transfer = architecture for making API

@app.get("/ping") # api
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict") # api
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

