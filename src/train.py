import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

# Chargement des données
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Sauvegarde du modèle
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model.joblib")

# Enregistrement du modèle dans Azure ML
workspace_name = "methoaml"
resource_group = "rg-open-ai"
subscription_id = "f80606e5-788f-4dc3-a9ea-2eb9a7836082"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name
)

registered_model = ml_client.models.create_or_update(
    Model(
        path=model_path,
        name="california-linear-model",
        description="Modèle de régression linéaire sur California Housing",
        type=AssetTypes.CUSTOM_MODEL,
        tags={"framework": "sklearn", "dataset": "california_housing"},
        properties={"mse": str(mse), "r2": str(r2)},
    )
)

print(f"✅ Modèle enregistré dans Azure ML avec l'ID : {registered_model.id}")
