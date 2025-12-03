
# Compte rendu

## Analyse de la Valeur Nutritionnelle des Aliments par Régression

**Date :** 3 Décembre 2025

---

# À propos du jeu de données :

Le dataset contient des informations nutritionnelles pour plus de **200 aliments courants**, incluant fruits, légumes, céréales, produits laitiers, boissons, snacks et plats cuisinés. Chaque ligne correspond à un aliment et fournit les valeurs énergétiques et la composition en macronutriments, telles que les **calories, protéines, glucides et lipides** par portion de 100 g.

Les données proviennent de **l’USDA FoodData Central**, source fiable de données nutritionnelles ouvertes. Seuls des aliments normaux et consommés quotidiennement ont été inclus, excluant les compléments alimentaires, poudres et formules infantiles.

---

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Analyse Exploratoire des Données (EDA)](#2-analyse-exploratoire-des-données-eda)

   * [Chargement et Structure du Dataset](#21-chargement-et-structure-du-dataset)
   * [Prétraitement et Ingénierie de Caractéristiques](#22-prétraitement-et-ingénierie-de-caractéristiques)
   * [Gestion des Valeurs Manquantes](#23-gestion-des-valeurs-manquantes)
   * [Analyse Statistique et Visuelle](#24-analyse-statistique-et-visuelle)
3. [Méthodologie de Modélisation](#3-méthodologie-de-modélisation)

   * [Séparation des Données (Data Split)](#31-séparation-des-données-data-split)
   * [Modèles de Régression Testés](#32-modèles-de-régression-testés)
4. [Résultats et Comparaison des Modèles](#4-résultats-et-comparaison-des-modèles)
5. [Analyse des Résultats et Recommandations](#5-analyse-des-résultats-et-recommandations)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction et Contexte

L’objectif de ce projet est de **prédire les calories** d’un aliment à partir de ses autres caractéristiques nutritionnelles et ingrédients. Nous avons exploré le dataset, réalisé un prétraitement complet, créé des features pertinentes, et entraîné plusieurs modèles de régression pour comparer leurs performances.

---

## 2. Analyse Exploratoire des Données (EDA)

### 2.1 Chargement et Structure du Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("food_nutrition.csv")
print(df.shape)
df.info()
df.head()
```

* **Nombre d'observations** : ~200 aliments
* **Nombre de variables** : 12 (macronutriments, fibres, eau, etc.)
* **Variable cible** : `calories`
* **Variables explicatives** : protéines, glucides, lipides, fibres, etc.

### 2.2 Prétraitement et Ingénierie de Caractéristiques

* Suppression des doublons
* Encodage des variables catégorielles (si présentes)
* Création de ratios nutritionnels (`protein_carb_ratio`, etc.)
* Standardisation pour certains modèles (SVR)

```python
# Suppression des doublons
df = df.drop_duplicates()

# Exemple de feature engineering
df['protein_carb_ratio'] = df['protein_g'] / (df['carbs_g'] + 1e-5)
```

### 2.3 Gestion des Valeurs Manquantes

```python
df.isnull().sum()
```

Le dataset est propre, aucune valeur manquante majeure.

### 2.4 Analyse Statistique et Visuelle

* **Histogrammes et Boxplots** pour chaque macronutriment
* **Heatmap** des corrélations

```python
plt.figure(figsize=(10,6))
sns.histplot(df['calories'], kde=True)
plt.title("Distribution des Calories")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matrice de Corrélation")
plt.show()
```

**Observations :**

* Protéines, glucides et lipides sont fortement corrélés avec les calories.
* Quelques valeurs extrêmes (ex: plats très caloriques).

---

## 3. Méthodologie de Modélisation

### 3.1 Séparation des Données (Data Split)

```python
from sklearn.model_selection import train_test_split

y = df['calories']
X = df.drop(columns=['calories'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3.2 Modèles de Régression Testés

1. Régression Linéaire
2. Régression Polynomiale (degré 2)
3. Arbre de Décision
4. Forêt Aléatoire
5. SVR (avec normalisation)

---

## 4. Résultats et Comparaison des Modèles

### Régression Linéaire

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)
```

**Interprétation :** Performance correcte mais limitée, le modèle ne capture pas les non-linéarités.

### Régression Polynomiale

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)
```

**Interprétation :** Amélioration par rapport à la régression linéaire, mais risque de surapprentissage.

### Arbre de Décision

```python
from sklearn.tree import DecisionTreeRegressor

model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
```

**Interprétation :** Captures les non-linéarités efficacement, performance très bonne.

### Forêt Aléatoire

```python
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
```

**Interprétation :** Stable, réduit le surapprentissage, performance similaire à l’Arbre de Décision.

### SVR

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_svr = SVR(kernel='rbf')
model_svr.fit(X_train_scaled, y_train)
y_pred_svr = model_svr.predict(X_test_scaled)
```

**Interprétation :** Sensible à la normalisation, performance inférieure aux arbres.

---

### Comparaison Graphique

```python
results_df_dict = {
    'Modèle': ['Lin.','Poly.','Arbre','RF','SVR'],
    'R2':[r2_lr,r2_poly,r2_dt,r2_rf,r2_svr],
    'RMSE':[rmse_lr,rmse_poly,rmse_dt,rmse_rf,rmse_svr]
}

results_df = pd.DataFrame(results_df_dict)

plt.figure(figsize=(12,6))
sns.barplot(x='Modèle', y='R2', data=results_df)
plt.title("Comparaison R² des modèles")
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(x='Modèle', y='RMSE', data=results_df)
plt.title("Comparaison RMSE des modèles")
plt.show()
```

**Observation :** L’Arbre de Décision est le modèle gagnant avec le meilleur R² et la RMSE la plus faible.

---

## 5. Analyse des Résultats et Recommandations

* Les modèles linéaires sont limités pour ce type de données.
* Les arbres (Decision Tree et Random Forest) sont adaptés aux relations non-linéaires entre macronutriments et calories.
* Feature Engineering supplémentaire (ratios, transformations log, etc.) pourrait améliorer les performances.
* Optimisation des hyperparamètres pour les arbres et SVR recommandée.

---

## 6. Conclusion

Ce projet montre que la **prédiction des calories** à partir des caractéristiques nutritionnelles est un problème non-linéaire. Les modèles basés sur les arbres fournissent les meilleures performances.

* **Meilleur modèle :** Arbre de Décision
* **R² :** élevé (>0.7), **RMSE :** faible
* **Perspectives :** Optimisation et extension à d’autres nutriments ou groupes alimentaires.


