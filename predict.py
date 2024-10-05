import pandas as pd
import pickle

# Cargar el modelo entrenado
with open('./models/my_ensemble_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Cargar el dataset
data = pd.read_csv('./data/credit_predictions_input.csv')

# Realizar predicciones (asumimos que todas las columnas son características para el modelo)
predictions = model.predict(data)

# Añadir la columna 'prediction' con las predicciones
data['prediction'] = predictions

# Guardar el nuevo dataset con la columna 'prediction'
data.to_csv('credit_predictions_output.csv', index=False)

print("Predicciones guardadas en 'credit_predictions_output.csv'")
