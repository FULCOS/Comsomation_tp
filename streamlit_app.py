import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Prédiction de la consommation électrique")

# Charger les données depuis un fichier CSV
df = pd.read_csv("donnees.csv", encoding="ISO-8859-1", sep=",")

# Transformer la colonne Date_Heure en une série temporelle et l'utiliser comme index
df['Date_Heure'] = pd.to_datetime(df['Date_Heure'], format='%Y/%m/%d %H:%M')
df.set_index('Date_Heure', inplace=True)

# Remplacer les données manquantes par la moyenne de chaque colonne
df = df.fillna(df.mean())

# Sélectionner les variables d'entrée et de sortie
X = df[['TempÃ©rature (Â°C)', 'Point de rosÃ©e', 'datehour', 'datemonth']]
y = df['consommation']

# Fractionner les données en ensembles de formation et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Créer un modèle XGBoost
model = XGBRegressor(learning_rate=0.1, n_estimators=150, max_depth=4, subsample=0.8, colsample_bytree=1.0)

# Former le modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Prédire la consommation électrique sur les données de test
y_pred = model.predict(X_test)

# Calculer la racine de l'erreur quadratique moyenne (RMSE) pour évaluer les performances du modèle
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write("RMSE:", rmse)
r2 = r2_score(y_test, y_pred)
st.write('R²:', r2)

# Prédire la consommation électrique pour les x prochaines lignes
last_date = df.index[-1]
next_dates = pd.date_range(start=last_date, periods=540, freq='3H')[1:]
next_X = X_test.iloc[-539:].values
next_y_pred = model.predict(next_X)

# Créer un graphique de la consommation prédite en fonction de la date
xfin = X.tail(50)
yfin = y.tail(50)

fig, ax = plt.subplots()
ax.plot(xfin.index, yfin, label='Données réelles')
ax.plot(next_dates, next_y_pred, label='Prédictions')
ax.set_xlabel('Date')
ax.set_ylabel('Consommation Prédite')
ax.set_title('Consommation Prédite vs. Date')
ax.legend()

# Afficher le graphique dans Streamlit
st.pyplot(fig)
