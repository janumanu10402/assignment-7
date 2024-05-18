from fastapi import FastAPI, UploadFile, File
from typing_extensions import Annotated
from keras.models import Sequential
from pydantic import BaseModel
import keras
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from typing import Callable
import math
import uvicorn

# Importing Prometheus modules for instrumentation
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import Info
from prometheus_client import Counter, Gauge

# Function to count API requests from each client IP
def api_client_requests_total() -> Callable[[Info], None]:
    api_client_requests_total = Counter(
    'api_client_requests_total',
    'Total number of API requests from each client IP address',
    ['client_ip']
)

    def instrumentation(info: Info) -> None:
        # Increment the counter with the client IP address as a label
        client_ip = info.request.client.host
        api_client_requests_total.labels(client_ip=client_ip).inc()
    return instrumentation

# Function to measure performance metrics
def measure_performance() -> Callable[[Info], None]:
    input_length_gauge = Gauge('input_length', 'Length of the images')
    total_time_gauge = Gauge('total_time', 'Total time taken by the API')
    procdeessing_time_per_char_gauge = Gauge('processing_time_per_char', 'Time per character',['client_ip'])

    async def instrumentation(info: Info) -> None:
        # Increment the counter with the client IP address as a label
        client_ip = info.request.client.host
        length = 784
        input_length_gauge.set(length)
        total_time_gauge.set(info.modified_duration*math.pow(10,6))
        procdeessing_time_per_char_gauge.labels(client_ip=client_ip).set((info.modified_duration*math.pow(10,6)/length))
    return instrumentation

# Path to the Keras model
path = "./model.keras"

# Helper function to load the Keras model
def load_model(path:str) -> Sequential:
    model = keras.models.load_model(path)
    return model

# Helper function to convert image to numpy array
def load_image_into_numpy_array(data):
    data1 = BytesIO(data)
    return np.array(Image.open(data1))

# Helper function to run the model and predict the digit
def predict_digit(model:Sequential, data_point:list) -> str:
    probs = model.predict(data_point, verbose=True)
    print("Predicted Digit:", np.argmax(probs))
    return str(np.argmax(probs))

# Helper function to format the received image
def format_image(data) -> list:
    data1 = BytesIO(data)
    image = Image.open(data1)
    image = image.resize((28,28)).convert("L") # Resize image to 28x28 pixels and convert to grayscale
    return np.array(image)

# Initialize FastAPI
app = FastAPI()

# Test route to ensure server is running
@app.get("/")
async def root():
    return {"message": "Hello World"}

# API endpoint for digit prediction, supports POST request
@app.post("/predict")
async def predict(image:UploadFile = File(...)):
    model = load_model(path) 
    formatted_image = format_image(await image.read())
    formatted_image = formatted_image.reshape(1,784) # Serialize into a 1-D array 
    digit = predict_digit(model, formatted_image)
    return {"digit": digit}

# Instrumentation and exposure of the app with Prometheus metrics
Instrumentator().add(api_client_requests_total()).add(measure_performance()).instrument(app).expose(app)

# Run the app with uvicorn server
if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
