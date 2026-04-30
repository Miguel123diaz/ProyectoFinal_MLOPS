import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier

# Cargar configuración
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Cargar dataset de entrenamiento
df = pd.read_csv(config["train_path"])

# Preprocesamiento
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Separar target
y = df[config["target_column"]]
X = df.drop(columns=[config["target_column"]])

# Convertir categóricas
X = pd.get_dummies(X, drop_first=True)

# Entrenar modelo
model = RandomForestClassifier(
    n_estimators=config["model"]["n_estimators"],
    max_depth=config["model"]["max_depth"],
    random_state=config["random_state"]
)
model.fit(X, y)

# Guardar modelo y columnas
joblib.dump(model, "models/model.pkl")
joblib.dump(X.columns, "models/columns.pkl")

print("✅ Modelo entrenado y guardado en models/")