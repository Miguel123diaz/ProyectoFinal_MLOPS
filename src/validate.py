import pandas as pd
import yaml
import joblib

# Cargar configuración
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Cargar dataset de prueba
df_test = pd.read_csv(config["dataset_path"])

# Preprocesamiento
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())
df_test["Embarked"] = df_test["Embarked"].fillna(df_test["Embarked"].mode()[0])
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())

# Separar target antes de convertir categóricas
y_test = df_test[config["target_column"]]
X_test = df_test.drop(columns=[config["target_column"]])

# Convertir categóricas a numéricas
X_test = pd.get_dummies(X_test, drop_first=True)

# Asegurar que las columnas coincidan con las del entrenamiento
train_columns = joblib.load("models/columns.pkl")
X_test = X_test.reindex(columns=train_columns, fill_value=0)

# Cargar modelo entrenado
model = joblib.load("models/model.pkl")

# Validar
accuracy = model.score(X_test, y_test)
print(f"✅ Accuracy en validación: {accuracy:.4f}")