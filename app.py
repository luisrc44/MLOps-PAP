import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def load_data(file_path):
    """Carga el dataset desde un archivo CSV"""
    data = pd.read_csv(file_path)
    X = data.drop('Y', axis=1)
    y = data['Y']
    return X, y

def scale_data(X_train, X_test):
    """Escala los datos usando StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def build_models():
    """Define los modelos a utilizar en el VotingClassifier"""
    
    # Modelo XGBoost
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=1,  
        n_estimators=500,
        max_depth=3,
        learning_rate=0.01,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=1234
    )

    # GradientBoostingClassifier
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=1234
    )

    # KNeighborsClassifier
    knn_model = KNeighborsClassifier(
        n_neighbors=5
    )

    # Support Vector Classifier (SVC)
    svc_model = SVC(
        probability=True,
        kernel='linear',
        C=1.0,
        random_state=1234
    )

    # Ensamble con 4 modelos
    ensemble_model = VotingClassifier(
        estimators=[('xgb', xgb_model), ('gb', gb_model), ('svc', svc_model), ('knn', knn_model)], 
        voting='soft'
    )
    
    return ensemble_model


def evaluate_model(model, X_train, X_test, y_train, y_test, threshold=0.45):
    """Evalúa el modelo, ajustando el umbral de decisión"""
    # Predicciones con probabilidades
    probas_train = model.predict_proba(X_train)[:, 1]
    y_hat_train = (probas_train > threshold).astype(int)

    probas_test = model.predict_proba(X_test)[:, 1]
    y_hat_test = (probas_test > threshold).astype(int)

    # Métricas
    f1_train = f1_score(y_train, y_hat_train)
    f1_test = f1_score(y_test, y_hat_test)
    accuracy_train = accuracy_score(y_train, y_hat_train)
    accuracy_test = accuracy_score(y_test, y_hat_test)

    # Resultados
    print(f"F1 Score Train: {f1_train:.2f}")
    print(f"F1 Score Test: {f1_test:.2f}")
    print(f"Accuracy Train: {accuracy_train:.2f}")
    print(f"Accuracy Test: {accuracy_test:.2f}")
    print("\nClasificación detallada:\n", classification_report(y_test, y_hat_test))
    print(f"ROC AUC: {roc_auc_score(y_test, probas_test):.2f}")

def main():
    # Cargar los datos
    X, y = load_data("credit_train.csv")

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

    # Escalar los datos
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Construir y entrenar el modelo de ensamble
    ensemble_model = build_models()
    ensemble_model.fit(X_train_scaled, y_train)

    # Evaluar el modelo
    evaluate_model(ensemble_model, X_train_scaled, X_test_scaled, y_train, y_test)

if __name__ == "__main__":
    main()
