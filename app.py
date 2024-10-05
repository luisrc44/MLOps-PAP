import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

app = FastAPI()

# Cargar el modelo y el escalador
model = pickle.load(open('./models/ensemble_model.pkl', 'rb'))
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))

# Ruta de saludo
@app.get("/")
def greet(name: str):
    return {
        "message": f"Hello, {name}!"
    }

# Verificación de salud del servidor
@app.get("/health")
def health_check():
    return {
        "status": "OK"
    }

# Ruta de predicción
@app.post("/predict")
def predict(data: list[float]):
    # Crear un DataFrame a partir de los datos de entrada
    X = [{
        f"X{i+1}": x
        for i, x in enumerate(data)
    }]
    
    df = pd.DataFrame.from_records(X)

    # Escalar los datos
    df_scaled = scaler.transform(df)

    # Realizar la predicción
    prediction = model.predict(df_scaled)
    
    # Guardar los nuevos puntos de datos en un archivo CSV para futuras actualizaciones del dataset
    df['prediction'] = prediction
    df.to_csv('new_data.csv', mode='a', header=False, index=False)

    return {
        "prediction": int(prediction[0])
    }

# Aplicar pruebas estadísticas
@app.post("/stat_tests")
def stat_tests(data1: list[float], data2: list[float]):
    # Convertir los datos de entrada en arrays de numpy para las pruebas estadísticas
    data1 = np.array(data1)
    data2 = np.array(data2)

    # Prueba Chi-cuadrado
    chi2_stat, p_val_chi2, _, _ = chi2_contingency([data1, data2])
    
    # Prueba de Kolmogorov-Smirnov
    ks_stat, p_val_ks = ks_2samp(data1, data2)

    chi2_result = "Chi-squared test passed" if p_val_chi2 >= 0.05 else "Chi-squared test failed"
    ks_result = "KS test passed" if p_val_ks >= 0.05 else "KS test failed"

    # Devolver los resultados de las pruebas
    return {
        "chi2_stat": chi2_stat,
        "p_val_chi2": p_val_chi2,
        "chi2_result": chi2_result,
        "ks_stat": ks_stat,
        "p_val_ks": p_val_ks,
        "ks_result": ks_result
    }

if __name__ == "__main__":
    uvicorn.run("app:app", port=1234, reload=True)
