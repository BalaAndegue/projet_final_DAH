from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Prédiction ventes supermarché")

# Chargement
model = joblib.load('model/supermarket_sales_model.pkl')
le_dict = joblib.load('model/label_encoders.pkl')  # ou 'label_encoders.pkl'

class Transaction(BaseModel):
    Branch: str
    City: str
    Customer_type: str
    Gender: str
    Product_line: str
    Unit_price: float
    Quantity: int
    Month: int
    Hour: int
    Payment: str
    Tax: float = None

@app.get("/")
def home():
    return {"message": "API prête ! Va sur /docs pour tester"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()

    # Mapping précis : champ du JSON → clé dans le_dict → nom final de colonne (exact comme dans le training)
    cat_mapping = [
        ("Branch", "Branch", "Branch_encoded"),
        ("City", "City", "City_encoded"),
        ("Customer_type", "Customer type", "Customer type_encoded"),   # ← espace !
        ("Gender", "Gender", "Gender_encoded"),
        ("Product_line", "Product line", "Product line_encoded"),     # ← espace !
        ("Payment", "Payment", "Payment_encoded")
    ]

    for field_name, dict_key, encoded_col_name in cat_mapping:
        value = data.pop(field_name)
        try:
            encoded_value = le_dict[dict_key].transform([value])[0]
            data[encoded_col_name] = encoded_value  # ← nom exact avec espace si besoin
        except ValueError as e:
            if "unseen labels" in str(e):
                return {"error": f"Valeur invalide pour {field_name}: '{value}'. Valeurs attendues: {list(le_dict[dict_key].classes_)}"}
            raise e

    # Ajouter les colonnes numériques avec les noms EXACTS du training
    data['Unit price'] = data.pop('Unit_price')  # ← espace
    data['Quantity'] = data.pop('Quantity')
    data['Month'] = data.pop('Month')
    data['Hour'] = data.pop('Hour')
    if 'Tax' in data:
        data['Tax'] = data.pop('Tax')

    # Créer le DataFrame avec les colonnes dans n'importe quel ordre (le modèle s'en fiche tant que les noms matchent)
    input_df = pd.DataFrame([data])

    # Prédiction
    prediction = model.predict(input_df)[0]

    return {"predicted_total_sales": round(prediction, 2)}