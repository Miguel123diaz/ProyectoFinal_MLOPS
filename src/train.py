import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar configuración desde config.yaml
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Leer dataset
df = pd.read_csv(config["dataset_path"])

# 3. Seleccionar columnas útiles (quitamos PassengerId, Name, Ticket, Cabin)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = config["target_column"]

# 4. Preprocesamiento básico
# Convertir variables categóricas a numéricas
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

# Rellenar valores faltantes
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

X = df[features]
y = df[target]

# 5. Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"], random_state=config["random_state"]
)

# 6. Entrenar modelo
model = RandomForestClassifier(
    n_estimators=config["model"]["n_estimators"],
    max_depth=config["model"]["max_depth"],
    random_state=config["random_state"]
)

# 7. MLflow tracking
mlflow.start_run()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

mlflow.log_param("n_estimators", config["model"]["n_estimators"])
mlflow.log_param("max_depth", config["model"]["max_depth"])
mlflow.log_metric("accuracy", acc)

mlflow.sklearn.log_model(model, "model")

mlflow.end_run()

print(f"Accuracy: {acc:.4f}")

