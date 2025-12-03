ikram el-wali
# Compte rendu

## Analyse de la Valeur Nutritionnelle des Aliments par Régression

**Date :** 3 Décembre 2025

---

# À propos du jeu de données :

Le dataset contient plus de **200 aliments courants**, chacun avec des informations nutritionnelles détaillées : **calories, protéines, glucides, lipides, fibres, eau, etc.** par portion de 100 g.

Ces données proviennent de **sources fiables ouvertes**, comme USDA FoodData Central, et représentent des aliments consommés quotidiennement. L’objectif est de prédire la **valeur calorique** à partir des autres caractéristiques nutritionnelles.

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
4. [Résultats et Analyse Détaillée](#4-résultats-et-analyse-détaillée)
5. [Analyse des Résultats et Recommandations](#5-analyse-des-résultats-et-recommandations)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction et Contexte

Le projet consiste à **prédire le nombre de calories** par portion d’un aliment à partir de ses caractéristiques nutritionnelles.
Nous avons suivi un pipeline complet : exploration des données, prétraitement, ingénierie de features, entraînement de modèles variés et comparaison des performances.

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

* **Observations :** 200 aliments
* **Variables :** 12 features + 1 cible (`calories`)
* **Cible :** `calories`
* **Features :** `protein_g`, `carbs_g`, `fat_g`, `fiber_g`, `water_g`, etc.

### 2.2 Prétraitement et Ingénierie de Caractéristiques

* Suppression des doublons : aucune entrée répétée inutile
* Création de **ratios** nutritionnels pour mieux capturer les relations non-linéaires
* Normalisation pour modèles sensibles à l’échelle (SVR)

```python
df = df.drop_duplicates()
df['protein_carb_ratio'] = df['protein_g'] / (df['carbs_g'] + 1e-5)
```

**Interprétation :** Le ratio protéines/glucides permet de détecter si un aliment est riche en protéines par rapport aux glucides, ce qui influence directement l’énergie.

### 2.3 Gestion des Valeurs Manquantes

```python
df.isnull().sum()
```

Aucune valeur manquante majeure : dataset prêt pour la modélisation.

### 2.4 Analyse Statistique et Visuelle

#### Histogramme des calories

```python
plt.figure(figsize=(10,6))
sns.histplot(df['calories'], kde=True)
plt.title("Distribution des Calories")
plt.show()
```

**Interprétation :**

* Les calories sont **skewed à droite**, avec quelques aliments très caloriques (plats cuisinés, snacks)
* La majorité des aliments (fruits, légumes, produits laitiers) ont moins de 500 kcal/100g.

#### Heatmap des corrélations

```python
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matrice de Corrélation")
plt.show()
```

**Observations :**

* Calories corrélées fortement avec **lipides, protéines et glucides**
* Faible corrélation avec l’eau et les fibres
* Ces relations justifient l’utilisation de modèles capables de capturer la non-linéarité.

---

## 3. Méthodologie de Modélisation

### 3.1 Séparation des Données (Data Split)

```python
from sklearn.model_selection import train_test_split

y = df['calories']
X = df.drop(columns=['calories'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Interprétation :**

* 80% pour entraînement, 20% pour test
* Random_state=42 pour reproductibilité

### 3.2 Modèles de Régression Testés

1. **Régression Linéaire** : baseline
2. **Régression Polynomiale** : capture interactions et non-linéarités
3. **Arbre de Décision** : modèle non-paramétrique
4. **Random Forest** : ensemble d’arbres pour réduction de variance
5. **SVR** : sensible à l’échelle, capture non-linéarité via noyau RBF

---

## 4. Résultats et Analyse Détaillée

### 4.1 Régression Linéaire

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

**Interprétation détaillée :**

* $R^2$ faible → les relations sont **non-linéaires**
* MSE élevé → les prédictions sur les aliments très caloriques sont peu précises
* Graphique **Réel vs Prédit** montre des points dispersés, surtout pour les valeurs extrêmes.

### 4.2 Régression Polynomiale

* Capture mieux les interactions comme **protéines × lipides**, mais surapprentissage possible
* Graphique Réel vs Prédit : points plus proches de la diagonale centrale, amélioration des calories moyennes, encore imprécis sur extrêmes.

### 4.3 Arbre de Décision

* $R^2$ très élevé (>0.7), RMSE faible
* Graphique Réel vs Prédit : presque tous les points alignés sur la diagonale → modèle excellent pour **plage complète de calories**
* **Analyse des résidus** : faible biais, variance réduite

### 4.4 Random Forest

* Stable, moins sensible aux valeurs extrêmes
* Légèrement inférieur à l’Arbre simple sur ce dataset spécifique (taille modeste)
* Graphique Réel vs Prédit : dispersion faible mais légèrement plus concentrée autour de la moyenne

### 4.5 SVR

* Sensible à la normalisation
* N’a pas capturé parfaitement la non-linéarité complexe
* Graphique Réel vs Prédit : dispersion visible sur les aliments très caloriques

### Comparaison des modèles (Graphique et Tableau)

```python
results_df_dict = {
    'Modèle': ['Lin.','Poly.','Arbre','RF','SVR'],
    'R2':[r2_lr,r2_poly,r2_dt,r2_rf,r2_svr],
    'RMSE':[rmse_lr,rmse_poly,rmse_dt,rmse_rf,rmse_svr]
}
results_df = pd.DataFrame(results_df_dict)
```

* Barplots R² : **Arbre > RF > Poly > Lin ≈ SVR**
* Barplots RMSE : **Arbre le plus bas, Lin et SVR les plus élevés**
* Interprétation : Les modèles arborescents capturent mieux les non-linéarités et interactions.

---

## 5. Analyse des Résultats et Recommandations

* **Arbre de Décision** : meilleur modèle, performance élevée sur toutes les plages de calories
* **Random Forest** : plus robuste pour généralisation, peut être améliorée par tuning des hyperparamètres
* **SVR et Linéaire** : insuffisants pour ce dataset non-linéaire
* **Recommandations :**

  1. Optimiser hyperparamètres de l’Arbre/Random Forest
  2. Créer de nouvelles features (ratios, combinaisons, transformations log)
  3. Étendre le modèle à d’autres nutriments (lipides, glucides, protéines)

---

## 6. Conclusion

* Relations **fortement non-linéaires** entre nutriments et calories
* Modèles linéaires sous-performent
* **Arbre de Décision** émerge comme modèle le plus efficace
* Possibilités d’amélioration via tuning, ensemble methods et feature engineering

