import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

# 1. Cargar configuración
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Cargar dataset de prueba (usamos test.csv de Kaggle)
df_test = pd.read_csv("data/test.csv")

# Seleccionar las mismas columnas que en train.py
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Preprocesamiento igual que en train.py
df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})
df_test["Embarked"] = df_test["Embarked"].map({"C": 0, "Q": 1, "S": 2})

df_test["Age"].fillna(df_test["Age"].median(), inplace=True)
df_test["Embarked"].fillna(df_test["Embarked"].mode()[0], inplace=True)
df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)

X_test = df_test[features]

# 3. Cargar el modelo desde MLflow
model_uri = "runs:/{}/model".format(mlflow.last_active_run().info.run_id)
model = mlflow.sklearn.load_model(model_uri)

# 4. Hacer predicciones
y_pred = model.predict(X_test)

print("Predicciones sobre test.csv:")
print(y_pred[:20])  # mostramos las primeras 20

