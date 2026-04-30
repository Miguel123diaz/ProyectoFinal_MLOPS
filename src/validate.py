import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Cargar configuración
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Cargar dataset de entrenamiento (porque test.csv no tiene Survived)
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

# Dividir en train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=config["random_state"]
)

# Asegurar columnas
train_columns = joblib.load("models/columns.pkl")
X_val = X_val.reindex(columns=train_columns, fill_value=0)

# Cargar modelo
model = joblib.load("models/model.pkl")

# Validar
accuracy = model.score(X_val, y_val)
y_pred = model.predict(X_val)

print(f"✅ Accuracy en validación: {accuracy:.4f}")
print("📊 Reporte de clasificación:")
print(classification_report(y_val, y_pred))