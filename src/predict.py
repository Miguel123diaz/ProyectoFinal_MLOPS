import pandas as pd
import joblib
import yaml

# Cargar configuración
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Cargar dataset de prueba
df_new = pd.read_csv(config["test_path"])

# Preprocesamiento igual que en entrenamiento/validación
df_new["Age"] = df_new["Age"].fillna(df_new["Age"].median())
df_new["Embarked"] = df_new["Embarked"].fillna(df_new["Embarked"].mode()[0])
df_new["Fare"] = df_new["Fare"].fillna(df_new["Fare"].median())

# Guardar IDs de pasajeros para referencia
passenger_ids = df_new["PassengerId"]

# Eliminar la columna target si existe (porque en predicción no debería estar)
if config["target_column"] in df_new.columns:
    df_new = df_new.drop(columns=[config["target_column"]])

# Convertir categóricas
X_new = pd.get_dummies(df_new, drop_first=True)

# Reindexar columnas para que coincidan con las del entrenamiento
train_columns = joblib.load("models/columns.pkl")
X_new = X_new.reindex(columns=train_columns, fill_value=0)

# Cargar modelo entrenado
model = joblib.load("models/model.pkl")

# Hacer predicciones
predictions = model.predict(X_new)

# Guardar resultados en CSV
output = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Prediction": predictions
})
output.to_csv("predictions.csv", index=False)

print("✅ Predicciones generadas y guardadas en predictions.csv")
