# Projet ML : Analyse et Prédiction des Ventes d’un Supermarché

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/BalaAndegue/supermarket-sales-predictor)
[![Render API](https://img.shields.io/badge/Render-API-green)](https://your-render-url.onrender.com) <!-- Remplace par ton URL Render si tu l'as -->
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Description du projet

Ce projet consiste en une **analyse complète des ventes d’un supermarché** à partir d’un dataset réel contenant 1000 transactions.  

Objectifs réalisés :
- Analyse exploratoire des données (EDA) : tendances, pics de ventes, catégories populaires, comportement clients
- Prédiction du montant total des ventes par transaction à l’aide d’un modèle de Machine Learning
- Déploiement d’une **API REST** (FastAPI) et d’une **interface web interactive** (Gradio)

Le modèle est capable de prédire avec une précision très élevée le montant total d’une transaction en fonction des caractéristiques (succursale, catégorie produit, client, etc.).

## 🛠️ Technologies utilisées

- **Python** (pandas, numpy, matplotlib, seaborn)
- **Scikit-learn** (RandomForestRegressor + LabelEncoder)
- **FastAPI** → API REST déployée sur Render
- **Gradio** → Interface web interactive déployée sur Hugging Face Spaces
- **Joblib** → Sauvegarde du modèle et des encodeurs
- **Hugging Face Spaces** & **Render.com** → Déploiement gratuit

## 🚀 Fonctionnalités

### Analyse Exploratoire (EDA)
- Ventes par ville, catégorie de produit, heure, genre, type de client
- Identification des pics saisonniers et horaires
- Visualisations claires (graphiques matplotlib/seaborn)

### Modèle de Prédiction
- RandomForestRegressor (R² ≈ 1.0 grâce à la relation déterministe `Total = Quantity × Unit price × 1.05`)
- Prédiction du montant total des ventes par transaction

### Démonstration Interactive
Lien : [https://huggingface.co/spaces/BalaAndegue/supermarket-sales-predictor](https://huggingface.co/spaces/BalaAndegue/supermarket-sales-predictor)

Interface intuitive avec :
- Menus déroulants pour les variables catégorielles
- Sliders pour prix, quantité, mois, heure
- Résultat instantané après clic sur "Prédire"

### API REST (FastAPI)
- Endpoint `/predict` acceptant un JSON avec les caractéristiques de la transaction
- Retourne le montant prédit
- Documentation interactive Swagger : `/docs`

## 📊 Dataset

Source : [Supermarket Sales - Kaggle](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales)

- ~1000 transactions
- Colonnes : Invoice ID, Branch, City, Customer type, Gender, Product line, Unit price, Quantity, Tax, Total, Date, Time, Payment, etc.

## 🏗️ Structure du projet

```
.
├── data
│   └── supermarket_sales.csv
├── main.py
├── model
│   ├── label_encoders.pkl
│   └── supermarket_sales_model.pkl
├── proposal.ipynb
├── __pycache__
│   └── main.cpython-312.pyc
├── README.md
└── requirements.txt

```

## 🔧 Installation locale (optionnel)

```bash
git clone https://github.com/BalaAndegue/projet_final_DAH.git
cd projet_final_DAH

pip install -r requirements.txt
```

### Lancer l'interface Gradio localement
```bash
python app.py
```

### Lancer l'API FastAPI localement
```bash
uvicorn main:app --reload
```

## 🌐 Déploiements

- **Interface web** : [Hugging Face Spaces](https://huggingface.co/spaces/BalaAndegue/supermarket-sales-predictor)
- **API REST** : Déployée sur Render (lien à ajouter si disponible)

## 📈 Résultats du modèle

- **MAE** : ~0.5 - 1.0
- **R²** : > 0.999 (précision quasi-parfaite due à la nature déterministe des données)

## 🙌 Auteur

**Bala Andegue**  
Étudiant en Data Science / Machine Learning

N’hésite pas à me contacter pour toute question ou collaboration !

---

⭐ Si ce projet vous plaît, n’hésitez pas à mettre une étoile au repo GitHub (à créer) !  

Merci d’avoir lu jusqu’ici ! 🛒💰