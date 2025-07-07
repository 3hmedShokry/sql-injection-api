from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# إعداد FastAPI
app = FastAPI(title="SQL Injection Detection API")

# تحميل النموذج المدرب
model = load_model("bilstm_model.h5")

# تحميل Tokenizer الحقيقي اللي اتدرب عليه الموديل
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# إعدادات التسلسل
max_sequence_length = 100  # يجب أن تطابق نفس القيمة وقت التدريب

# هيكل البيانات المستلمة
class InputData(BaseModel):
    sentence: str

# نقطة النهاية للتنبؤ
@app.post("/predict")
async def predict(data: InputData):
    sentence = data.sentence

    # تحويل الجملة إلى تسلسل أرقام + padding
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')

    # التنبؤ من الموديل
    prediction = model.predict(np.array(padded))
    label = int(prediction[0][0] > 0.7)

    return {
        "sentence": sentence,
        "prediction": label,       # 1 = SQLi, 0 = Safe
        "confidence": float(prediction[0][0])
    }

# لتشغيل السيرفر مباشرة من الملف
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
