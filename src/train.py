import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Cargar configuración
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Cargar dataset
df = pd.read_csv(config["dataset_path"])

# Preprocesamiento simple
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Separar target antes de convertir categóricas
y = df[config["target_column"]]
X = df.drop(columns=[config["target_column"]])

# Convertir categóricas a numéricas
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"], random_state=config["random_state"]
)

# Entrenar modelo
model = RandomForestClassifier(
    n_estimators=config["model"]["n_estimators"],
    max_depth=config["model"]["max_depth"],
    random_state=config["random_state"]
)
model.fit(X_train, y_train)

# Registrar en MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("accuracy", model.score(X_test, y_test))

# Guardar modelo y columnas
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(X.columns, "models/columns.pkl")
print("✅ Modelo entrenado y guardado en models/model.pkl")