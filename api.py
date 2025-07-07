from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import requests

app = FastAPI(title="SQL Injection Detection API")

# Google Drive file IDs
MODEL_ID = "1IsPUNwRavK1MY_RCV1DrwuY9ibpTOIdW"
TOKENIZER_ID = "1YQah6VKWm3-Q3Hzo5Gcgezhp3sE5m1vc"

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Download model and tokenizer if not present
if not os.path.exists("bilstm_model.h5"):
    print("Downloading model...")
    download_file_from_google_drive(MODEL_ID, "bilstm_model.h5")

if not os.path.exists("tokenizer.pkl"):
    print("Downloading tokenizer...")
    download_file_from_google_drive(TOKENIZER_ID, "tokenizer.pkl")

model = load_model("bilstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_length = 100

class InputData(BaseModel):
    sentence: str

@app.post("/predict")
async def predict(data: InputData):
    sentence = data.sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')
    prediction = model.predict(np.array(padded))
    label = int(prediction[0][0] > 0.7)
    return {
        "sentence": sentence,
        "prediction": label,
        "confidence": float(prediction[0][0])
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)