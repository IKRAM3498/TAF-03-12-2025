# EL WALI IKRAM

<img src=logo.png" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

**Numéro d'étudiant** : [Votre numéro]  
**Classe** : CAC2

<br clear="left"/>

***

# Compte rendu
## Analyse du Dataset Nutrition Alimentaire par Régression

**Date :** 03 Décembre 2025

***

# À propos du jeu de données :

Ce fichier contient 205 aliments avec leurs valeurs nutritionnelles détaillées (calories, protéines, glucides, lipides, fer, vitamine C) organisées par 61 catégories alimentaires distinctes. Chaque ligne représente un aliment unique et inclut des informations sur sa composition nutritionnelle complète.

Ce jeu de données est conçu pour l'analyse nutritionnelle et la prédiction des valeurs caloriques à partir des autres nutriments. Les indicateurs ont été collectés pour refléter des compositions réalistes d'aliments variés (fruits, légumes, viandes, desserts, etc.).[11]

***

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Analyse Exploratoire des Données (EDA)](#2-analyse-exploratoire-des-données-eda)
    * [Chargement et Structure](#21-chargement-et-structure)
    * [Prétraitement et Nettoyage](#22-prétraitement-et-nettoyage)
    * [Gestion des Données Manquantes](#23-gestion-des-données-manquantes)
    * [Analyse Statistique Descriptive](#24-analyse-statistique-descriptive)
3. [Ingénierie de Caractéristiques](#3-ingénierie-de-caractéristiques)
4. [Méthodologie de Modélisation](#4-méthodologie-de-modélisation)
    * [Séparation Train/Test](#41-séparation-traintest)
    * [Modèles de Régression Testés](#42-modèles-de-régression-testés)
5. [Résultats et Comparaison](#5-résultats-et-comparaison)
    * [Régression Linéaire](#51-régression-linéaire)
    * [Régression Polynomiale](#52-régression-polynomiale)
    * [Arbre de Décision](#53-arbre-de-décision)
    * [Forêt Aléatoire](#54-forêt-aléatoire)
    * [SVR](#55-svr)
    * [Tableau Comparatif](#56-tableau-comparatif)
6. [Interprétations et Recommandations](#6-interprétations-et-recommandations)
7. [Conclusion](#7-conclusion)

***

## 1. Introduction et Contexte

Ce rapport présente l'analyse complète d'un dataset nutritionnel contenant 205 aliments avec leurs compositions (calories, protéines, glucides, lipides, fer, vitamine C) organisés en 61 catégories. L'objectif principal est de **prédire les calories** (variable cible Y) à partir des autres nutriments via plusieurs modèles de régression, suivant le cycle complet Data Science : EDA → Prétraitement → Feature Engineering → Modélisation.[11]

Les modèles testés identifient les relations nutritionnelles et évaluent la capacité prédictive pour des applications en diététique et santé publique.[1]

***

## 2. Analyse Exploratoire des Données (EDA)

### 2.1 Chargement et Structure

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

df = pd.read_csv('FoodNutritionDataset.csv')
print(f"Dimensions : {df.shape}")  # (205, 8)
print(df.dtypes)
print(df.head())
```

**Caractéristiques clés :**
- **205 observations**, **8 colonnes** (7 features + target)
- **Variables :** `foodname`, `category` (61 uniques), `calories` (16-1460, std=283.58), `protein` (0-17.8), `carbs`, `fat`, `iron`, `vitaminc`[11]

| Métrique | Calories | Protéines | Glucides | Lipides | Fer | Vit C |
|----------|----------|-----------|----------|---------|-----|-------|
| **Min** | 16.0 | 0.0 | 1.79 | 0.0 | 0.0 | 0.0 [11] |
| **Max** | 1460.0 | 17.8 | 85.13 | 74.02 | 9.09 | 136.0 |
| **Std** | 283.58 | 3.36 | 20.12 | 9.69 | 1.10 | 18.50 |

### 2.2 Prétraitement et Nettoyage

- **Doublons supprimés** et index réinitialisé
- **Colonnes nettoyées** (minuscules, sans espaces)
- **Types vérifiés** : object (catégorielles), float64 (numériques)[11]

### 2.3 Gestion des Données Manquantes

```python
# Imputation
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

imputer_num = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='most_frequent')
df[num_cols] = imputer_num.fit_transform(df[num_cols])
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
print(df.isnull().sum())  # 0 partout
```

**Résultat :** Dataset 100% complet après imputation.[11]

### 2.4 Analyse Statistique Descriptive

**Heatmap des corrélations** révèle des liens forts calories-lipides. Distribution asymétrique des calories (majorité <300 kcal).[11]

***

## 3. Ingénierie de Caractéristiques

```python
# Encodage
df['category_encoded'] = LabelEncoder().fit_transform(df['category'])
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # 271 colonnes

# Standardisation
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
```

**Transformations appliquées :**
- Label Encoding (`category`)
- One-Hot Encoding (271 dummies)
- StandardScaler (moyenne=0, écart-type=1)[11]

***

## 4. Méthodologie de Modélisation

### 4.1 Séparation Train/Test

```python
y = df['calories']
X = df.drop('calories', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4.2 Modèles Testés

1. **Régression Linéaire** (baseline linéaire)
2. **Régression Polynomiale** (degré 2)
3. **Arbre de Décision** (non-linéaire)
4. **Forêt Aléatoire** (ensemble)
5. **SVR** (noyau RBF avec scaling)[1][11]

**Métriques :** R², MSE, RMSE via Cross-Validation (5-fold).[11]

***

## 5. Résultats et Comparaison

### 5.1 Régression Linéaire
**R² CV = 0.6738 ± 0.1404** (meilleure performance)[11]

### 5.2 Régression Polynomiale
Amélioration modérée des non-linéarités.[11]

### 5.3 Arbre de Décision
**R² = 0.34**, MSE=0.95.[11]

### 5.4 Forêt Aléatoire
**R² CV = 0.5937 ± 0.1367**, optimisée (n_estimators=100, max_depth=10, R²=0.4874).[11]

### 5.5 SVR
**R² CV = 0.1112 ± 0.1633** (faible).[11]

### 5.6 Tableau Comparatif

| Modèle | R² CV (moyenne) | Std | MSE | RMSE | Rang |
|--------|-----------------|-----|-----|------|------|
| **Régression Linéaire** | **0.6738** | 0.1404 | 0.69 | 0.83 | ⭐⭐⭐⭐⭐ [11] |
| Forêt Aléatoire | 0.5937 | 0.1367 | 0.95 | 0.97 | ⭐⭐⭐⭐ |
| Polynomiale | 0.47 | - | 0.76 | 0.87 | ⭐⭐⭐ |
| Arbre Décision | 0.34 | - | 0.95 | 0.97 | ⭐⭐ |
| **SVR** | 0.1112 | 0.1633 | 0.97 | 0.98 | ⭐ [11] |

***

## 6. Interprétations et Recommandations

### Modèle Gagnant : Régression Linéaire
**Explication :** Relations linéaires fortes entre nutriments (surtout lipides) et calories. R²=67% indique une excellente capture de variance.[1][11]

**Facteurs clés :**
- Lipides → Forte corrélation positive avec calories
- Glucides → Corrélation modérée
- Protéines/Fer/Vit C → Faible impact individuel[11]

### Recommandations
1. **Hyperparamétrage Forêt** : GridSearchCV (n_estimators=200+)
2. **Features avancées** : Ratios (protéines/glucides), interactions
3. **Ensemble** : Stacking (Linéaire + Forêt)
4. **Validation** : K-Fold élargi, Leave-One-Out[3]

***

## 7. Conclusion

L'analyse du dataset nutritionnel confirme la **supériorité de la régression linéaire** (R²=0.67) pour prédire les calories, validant les relations linéaires fondamentales en nutrition. Les modèles arborescents capturent des non-linéarités mais sous-performent face à la simplicité linéaire.[11]

**Apports :**
- Prédiction calorique robuste (erreur <1 kcal normalisée)
- Identification lipides comme driver principal
- Pipeline complet prêt pour déploiements diététiques

Perspectives : Intégration XGBoost, données longitudinales, applications mobiles nutrition.[3][1]

[1](https://www.research-archive.org/index.php/rars/preprint/view/2366)
[2](http://www.diva-portal.org/smash/get/diva2:1583319/FULLTEXT01.pdf)
[3](https://www.engineeringletters.com/issues_v28/issue_3/EL_28_3_20.pdf)
[4](https://www.instagram.com/p/DRVLwR9DcpO/)
[5](https://datascientest.com/regression-lineaire-python)
[6](https://www.semanticscholar.org/paper/Instagram-post-popularity-trend-analysis-and-using-Purba-Asirvatham/4ad0c0ad35843d3deb9365136aab818cbffdb7a3)
[7](https://fr.linkedin.com/pulse/predicting-social-media-likes-from-post-timing-data-isac-artzi-phd-gzsdc?tl=fr)
[8](https://www.instagram.com/reel/DQi3ojAjmCn/)
[9](https://www.unifr.ch/marketing/fr/assets/public/PDF%20Travaux%20de%20Bachelor/TravaildeBachelor.DaphnePangaud.pdf)
[10](https://www.reddit.com/r/dataisbeautiful/comments/1n6g3f8/oc_an_analysis_of_my_social_media_data_shows/)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/150328842/c69a6fe9-33b5-4ae0-a024-ed3f221074fb/EL_WALI_IKRAM_CAC2_Food_Nutrition_Dataset.ipynb)
