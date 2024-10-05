# demo.py

import requests

# URL de tu API FastAPI
url = "http://localhost:1234/predict"

# Datos de entrada (asegúrate de que coincidan con las características de tu modelo)
data = list(range(23))  # Modifica esto según el número de características que tienes

# Realizar la solicitud POST a la API
response = requests.post(url, json={"features": data})

# Verificar que la solicitud fue exitosa
if response.status_code == 200:
    prediction = response.json().get("prediction")
    print(f"Predicción: {prediction}")
else:
    print(f"Error: {response.status_code} - {response.text}")
