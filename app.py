from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle  # Usaremos pickle en lugar de joblib para el manejo del modelo

# Inicializamos la aplicación FastAPI
app = FastAPI()

# Clase para recibir la entrada de datos
class DataInput(BaseModel):
    features: list

# Cargar el modelo entrenado
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Ruta para realizar predicciones
@app.post("/predict")
def predict(input_data: DataInput):
    # Convertir la entrada en un array de numpy
    input_array = np.array(input_data.features).reshape(1, -1)

    # Realizamos la predicción
    prediction = model.predict(input_array)

    # Devolver la predicción
    return {"prediction": int(prediction[0])}


import pickle

# Después de entrenar el modelo
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
