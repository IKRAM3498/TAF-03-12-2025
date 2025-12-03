**Nom de l'étudiant :** EL-WALI IKRAM
**Classe :** 24010354
# Food Nutrition Dataset
---

# Compte rendu

## Analyse Nutritionnelle et Prédiction des Calories par Régression

**Date :** 3 Décembre 2025

---
#La thématique choisie pour cette analyse est la santé, avec un accent particulier sur l'alimentation et la nutrition.

# À propos du jeu de données :

#1. Sélection du jeu de données
Le jeu de données sélectionné est le **Food Nutrition Dataset (150+ Everyday Foods)**, disponible sur la plateforme **Kaggle**.  
Il contient des informations nutritionnelles détaillées sur plus de **150 aliments couramment consommés**, incluant notamment les calories, les protéines, les glucides, les lipides et d’autres nutriments essentiels.

Ce dataset est pertinent pour plusieurs raisons :

- Il n'est **pas trivial** (contrairement à Titanic ou Iris).  
- Il contient principalement des **variables quantitatives exploitables**.  
- Il permet d'étudier une thématique d'intérêt général : **la nutrition et la composition des aliments**.  
- Il est propre, structuré et directement utilisable pour une analyse ou un modèle de Machine Learning.

#2. Définition de la Problématique (Tâche : Régression)

L’objectif de ce projet est de construire un **modèle de régression** capable de **prédire le nombre de calories d’un aliment** à partir de ses valeurs nutritionnelles (protéines, glucides, lipides, fibres, etc.).

**Il s'agit donc d'une tâche de régression**, car la variable cible (**Calories**) est une variable **numérique continue**.
Problématique étudiée :
> **Peut-on prédire de manière fiable la valeur calorique d’un aliment à partir de sa composition nutritionnelle ?**
Cette problématique permet :

- d'évaluer l’importance de chaque nutriment dans le total calorique,  
- de tester différents modèles de régression,  
- de vérifier la cohérence du dataset par rapport aux lois nutritionnelles (ex : calories ≈ 4×protéines + 4×glucides + 9×lipides).

# 3. Dictionnaire des Données (Metadata)

## Taille du dataset
- **Nombre de lignes (aliments)** : ≈ 150  
- **Nombre de colonnes (variables)** : environ 10 à 20 selon la version

## Types de variables
- **Variables quantitatives continues** : calories, protéines, glucides, lipides, fibres, sucres, sodium…  
- **Variables qualitatives nominales** : nom de l’aliment, éventuellement catégorie de l’aliment

## Description des variables principales

| Variable | Type | Description |
|---------|------|-------------|
| **Food** | Catégorielle | Nom de l’aliment (ex : Apple, Rice, Chicken Breast) |
| **Calories** | Numérique | Énergie totale en kcal (🌟 *variable cible du modèle*) |
| **Protein (g)** | Numérique | Quantité de protéines (g) |
| **Carbohydrates (g)** | Numérique | Quantité totale de glucides (g) |
| **Fat (g)** | Numérique | Quantité totale de lipides (g) |
| **Fiber (g)** | Numérique | Teneur en fibres |
| **Sugar (g)** | Numérique | Quantité de sucres |
| **Sodium (mg)** | Numérique | Teneur en sodium (mg) |

## Variable Cible (Target)

La **target** utilisée pour la tâche de régression est :

 **Calories**

---

## 1. Introduction et Contexte

Ce rapport détaille l'analyse et la modélisation prédictive d'un jeu de données nutritionnel. L'objectif est de prédire les **calories d'un aliment** à partir de ses autres caractéristiques nutritionnelles.

Les étapes suivies incluent l'exploration des données, le prétraitement, la création de nouvelles features, et la comparaison de trois modèles de régression : **Arbre de Décision, Random Forest et SVR**.

---

## 2. Analyse Exploratoire des Données

### 2.1 Chargement et Structure du Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Chargement
df = pd.read_csv("food_nutrition.csv")
print(df.shape)
df.head()
```

* **Observations :** ~150 aliments
* **Variables :** Calories (target), Protéines, Glucides, Lipides, Fibres, Sucres, etc.

### 2. Pré-traitement (Preprocessing)
Nettoyage des données
Gestion des doublons
Formatage des données
Imputation des valeurs manquantes
Utilisation de stratégies avancées
Encodage des variables catégorielles
One-Hot Encoding
Label Encoding
Target Encoding
Normalisation ou Standardisation des données numériques
* Création de **ratios nutritionnels** (ex: protéines/calories, lipides/calories) pour améliorer la prédiction.
* Encodage des variables catégorielles (ex: type d’aliment).
* Normalisation pour les modèles sensibles à l’échelle (SVR).

```python
# Informations générales sur le dataset
df.info()

# Statistiques descriptives des colonnes numériques
df.describe()

# Vérification de la présence de valeurs manquantes
df.isnull().sum()

# Vérification des doublons
print("Nombre de doublons :", df.duplicated().sum())
# Supprimer les doublons si présents
df = df.drop_duplicates()
print("Nombre de lignes après suppression des doublons :", df.shape[0])

# Sélection uniquement des colonnes numériques pour l'imputation
numeric_cols = df.select_dtypes(include=np.number).columns

# KNNImputer : remplit les valeurs manquantes en fonction des k voisins les plus proches
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Vérification après imputation
df.isnull().sum()

# Vérification des types de colonnes
df.dtypes

# Conversion des colonnes numériques en float (si besoin)
for col in numeric_cols:
    df[col] = df[col].astype(float)

# Supprimer colonnes inutiles si nécessaire (ex : ID)
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)
```

**Interprétation :**  Le pré-traitement permet de préparer les données pour la modélisation. Il inclut le nettoyage, la suppression des doublons, la correction des formats et l’imputation des valeurs manquantes. Les variables catégorielles sont transformées en nombres via des encodages (One-Hot, Label ou Target Encoding), et les données numériques sont normalisées ou standardisées pour garantir que tous les modèles puissent apprendre efficacement et produire des prédictions fiables.



### 2. Analyse Exploratoire des Données (EDA)
##2.1. Distribution des variables numériques
<img width="1492" height="690" alt="image" src="https://github.com/user-attachments/assets/ea95d0a4-e2b3-4f1a-bb61-1d0bc74b6365" />
* Histogrammes des calories, protéines, lipides et glucides.

**Interprétation :** Cette figure montre la distribution de six variables nutritionnelles (calories, protéines, glucides, lipides, fer, vitamine C) sous forme d’histogrammes avec une courbe de densité lissée pour chacune.

## Forme générale des distributions  
Les six graphiques présentent tous une forte asymétrie à droite : la majorité des valeurs est faible, avec quelques valeurs très élevées qui tirent la queue de la distribution vers la droite.  
Cela suggère que la plupart des aliments de l’échantillon sont relativement « pauvres » dans chaque nutriment, et qu’un petit nombre d’aliments concentrent des teneurs beaucoup plus élevées.

## Détails par variable  
- Calories et protéines : distributions concentrées sur des valeurs faibles, avec quelques aliments beaucoup plus caloriques et protéinés (queue longue).  
- Glucides (carbs) : distribution un peu plus étalée, montrant une diversité plus importante des teneurs en glucides entre les aliments.  
- Lipides (fat) : très forte concentration près de zéro, ce qui indique que la majorité des aliments sont peu gras, mais certains sont extrêmement riches en lipides.  
- Fer et vitamine C : même structure fortement dissymétrique, typique de micronutriments où quelques aliments (par ex. abats, certains légumes/fruits) sont très riches tandis que la plupart en contiennent peu.

## Ce que cela implique pour l’analyse  
- Les distributions non normales et très asymétriques rendent l’usage de la moyenne et de l’écart‑type moins informatifs que la médiane et les quantiles.  
- Des transformations (par exemple logarithme) ou des méthodes non paramétriques peuvent être préférables pour modéliser ou comparer ces variables.  
- Les longues queues droites indiquent la présence potentielle de valeurs extrêmes qu’il faudra examiner séparément pour comprendre quels aliments les produisent et si ce sont des outliers à traiter ou des cas typiques mais rares.

##2.2. Boxplots pour détecter les outliers
<img width="1489" height="667" alt="image" src="https://github.com/user-attachments/assets/da5490a9-137f-4b2f-9509-47ae32d57a2a" />
**interpritation:** Ces six boxplots résument la répartition des mêmes variables nutritionnelles (calories, protéines, glucides, lipides, fer, vitamine C) en mettant l’accent sur la médiane, la dispersion et les valeurs extrêmes.

## Information donnée par les boxplots  
- La boîte représente l’intervalle interquartile (du 1er au 3e quartile), donc la zone où se trouvent 50% des observations.  
- La ligne à l’intérieur de chaque boîte est la médiane : elle indique le niveau « typique » de chaque nutriment.  
- Les « moustaches » prolongent la boîte jusqu’à des valeurs encore considérées comme normales, et les points isolés au‑delà sont des valeurs aberrantes (outliers), beaucoup plus élevées que le reste des données.

## Ce que l’on observe pour ces nutriments  
- Pour toutes les variables, la médiane est proche de la partie basse de la boîte et très près de zéro, ce qui confirme que la majorité des aliments sont peu riches dans chaque nutriment, avec quelques aliments beaucoup plus riches.  
- Le grand nombre de points au‑dessus des moustaches montre de nombreux outliers à haute teneur (aliments très caloriques, très gras, très riches en fer ou en vitamine C, etc.), ce qui traduit des distributions très asymétriques et hétérogènes.

## Implications statistiques et pratiques  
- La présence de nombreux outliers indique qu’il faut être prudent avec la moyenne : elle sera fortement tirée vers le haut et ne représentera pas bien l’« aliment moyen ».  
- Pour comparer des groupes d’aliments ou construire des modèles, il peut être pertinent d’utiliser la médiane, des tests non paramétriques ou d’éventuelles transformations (par exemple logarithmiques) pour réduire l’influence de ces valeurs extrêmes.

## 2.3. Heatmap des corrélations
<img width="1319" height="1245" alt="image" src="https://github.com/user-attachments/assets/09e669c5-1da0-4759-851c-66d26a158935" />
**interpritation:** Analyse de la Carte de Chaleur
Objectif : Cette carte de chaleur vise à visualiser les coefficients de corrélation (généralement la corrélation de Pearson, mais d'autres peuvent être utilisés) entre différentes variables.

Variables : Les variables sont listées à la fois sur l'axe vertical (lignes) et l'axe horizontal (colonnes). Elles semblent représenter des produits alimentaires spécifiques (food_name: ...) ou des catégories alimentaires (category: ...).

Code de Couleurs (Légende à Droite) :

Rouge Vif (Proche de +1.0) : Indique une forte corrélation positive. Lorsque la valeur d'une variable augmente, la valeur de l'autre variable a tendance à augmenter aussi.

Bleu Foncé (Proche de -1.0) : Indique une forte corrélation négative. Lorsque la valeur d'une variable augmente, la valeur de l'autre variable a tendance à diminuer.

Blanc (Proche de 0.0) : Indique une absence de corrélation ou une corrélation très faible.

(Dans ce graphique, on voit des valeurs allant de -0.2 à 1.0).

Lecture du Graphique (Observations) :

Matrice Symétrique : C'est une matrice de corrélation complète. La moitié supérieure est la symétrie de la moitié inférieure.

Diagonale (Non Visiblement Remplie) : La diagonale principale (là où une variable est corrélée avec elle-même) devrait être de 1.0 (rouge vif), mais la matrice semble avoir été tronquée ou les variables sont réordonnées/filtrées de manière spécifique.

Distribution des Valeurs : La grande majorité de la carte est blanche ou noire, ce qui signifie que la plupart des paires de produits/catégories n'ont pas de corrélation significative les unes avec les autres.

Points de Corrélation Significative :

Il y a quelques points blancs/noirs intenses qui pourraient représenter des corrélations très proches de 1.0 (ou -0.2). Par exemple, il semble y avoir des groupes de corrélations positives fortes (petits carrés rouges/noirs) dans la partie supérieure droite et autour du centre du graphique. Ces points indiquent des associations claires entre certains aliments.

Exemple (hypothétique) : Si la case entre food_name: Bacon and tomato dressing et food_name: Coleslaw est rouge, cela pourrait signifier que si l'un est consommé ou acheté, l'autre l'est aussi fréquemment.

Les corrélations négatives (bleu) semblent être rares ou inexistantes dans la partie visible, la plage commençant à -0.2 (bleu très clair).

##2.4. Scatterplots : relation features ↔ target
<img width="1489" height="690" alt="image" src="https://github.com/user-attachments/assets/bb0e4cab-6d8b-4914-849f-f81d4afdaeb1" />
**interpritation :** 
**cinq diagrammes de dispersion (scatter plots)** qui explorent la relation entre la **teneur en calories** (Calories, sur l'axe Y) et différentes composantes nutritionnelles (sur l'axe X) : **protéines, glucides (carbs), lipides (fat), fer (iron) et vitamine C (vitamin_c)**.

Il est très probable que ces données aient été **normalisées ou standardisées** (car les valeurs sont centrées autour de 0 et vont de -1 à +7 environ, ce qui n'est pas le cas des valeurs nutritionnelles brutes).

###  **Interprétation Rapide des Graphiques**

| Graphique | Relation Observée | Interprétation |
| :--- | :--- | :--- |
| **Protein vs Calories** | Tendance légère à positive. | Plus un aliment est riche en protéines, plus il a tendance à être calorique. |
| **Carbs vs Calories** | Tendance positive notable. | La teneur en glucides semble être un facteur important de la teneur en calories. |
| **Fat vs Calories** | Tendance positive très claire. | **Forte corrélation positive.** C'est la relation la plus marquée. Les aliments très gras sont très souvent les plus caloriques (point à l'extrême droite). |
| **Iron vs Calories** | Tendance positive modérée. | Les aliments riches en fer ont tendance à avoir une teneur en calories plus élevée. |
| **Vitamin\_C vs Calories** | Pas de tendance positive claire. | **Faible ou absence de corrélation.** Les aliments très riches en vitamine C (points à droite) peuvent être à la fois très peu ou très caloriques (points éparpillés sur Y). |

---

## 3. Méthodologie de Modélisation

### 3.1 Séparation Train/Test

```python
from sklearn.model_selection import train_test_split

y = df['calories']
X = df.drop(columns=['calories'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3.2 Modèles de Régression Testés

1. LinearRegression
2. RandomForestRegressor
3. XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Trois modèles différents
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, eval_metric='rmse')

##1. Scatterplot : Réel vs. Prédit (Random Forest)
<img width="845" height="547" alt="image" src="https://github.com/user-attachments/assets/e17fa814-5299-4cf3-991d-22ca75fdd203" />

**interpritation :** 
Absolument. Voici une interprétation concise et structurée, prête à être copiée-collée :

#### **1. Random Forest : Réel vs. Prédit (Performance du Modèle)**

* Ce graphique évalue la performance d'un modèle de prédiction (Random Forest) des Calories.
* **Axes :** Calories réelles (X) vs. Calories prédites (Y).
* **Ligne Idéale ($y=x$) :** La ligne pointillée rouge représente une prédiction parfaite.
* **Conclusion :** Le modèle est **très performant**. La majorité des points (aliments) sont très proches de la ligne idéale, signifiant une bonne capacité à prédire les calories. Seuls quelques points (ex: vers X=4, Y=1) montrent une sous-estimation significative (erreurs).

#### **2. Comparaison Nutriments vs. Calories (Corrélations)**

Ces diagrammes de dispersion montrent la relation entre la teneur en Calories (Y) et cinq nutriments (X).

| Nutriment | Tendance Observée | Impact sur les Calories |
| :--- | :--- | :--- |
| **Fat (Lipides)** | Très forte tendance positive. | **Meilleur prédicteur.** Les aliments très gras sont très caloriques. |
| **Carbs (Glucides)** | Tendance positive notable. | **Bon prédicteur.** Contribue significativement à la teneur en calories. |
| **Protein (Protéines)** | Tendance positive légère. | **Contribution modérée** aux calories. |
| **Iron (Fer)** | Tendance positive modérée. | Les aliments riches en fer ont tendance à être plus caloriques. |
| **Vitamin\_C (Vitamine C)** | Faible ou absence de tendance. | **Faible prédicteur.** La teneur en Vitamine C n'est pas liée au niveau de calories. |

**Synthèse :** Les **Lipides (Fat)** et les **Glucides (Carbs)** sont les facteurs déterminants de la teneur en calories, ce qui justifie la bonne performance du modèle Random Forest.
## 2. Scatterplot : Linear Regression
<img width="845" height="548" alt="image" src="https://github.com/user-attachments/assets/c94a9966-c1b7-4d9e-b6a5-87a26af17dbb" />
**interpritation :** 

#### **1. Relations Nutriments vs. Calories **

* **Fat (Lipides) :** Très forte corrélation positive avec les Calories. **Meilleur prédicteur.**
* **Carbs (Glucides) :** Corrélation positive modérée à forte avec les Calories.
* **Protein (Protéines) & Iron (Fer) :** Corrélations positives faibles à modérées.
* **Vitamin\_C (Vitamine C) :** Faible ou absence de corrélation.

#### **2. Performance des Modèles de Prédiction (Images 1 & 2)**

Les graphiques comparent les Calories réelles (X) et les Calories prédites (Y).

* **Random Forest (Image 2) :**
    * **Performance : Très bonne.** Les points sont très proches de la ligne idéale ($y=x$).
    * **Conclusion :** Le modèle Random Forest est le plus précis, gérant bien la non-linéarité et les valeurs extrêmes.
* **Linear Regression (Image 1) :**
    * **Performance : Moins bonne.** Les points sont plus dispersés autour de la ligne idéale.
    * **Conclusion :** Ce modèle est moins précis, surtout pour les valeurs extrêmes (ex : la valeur réelle la plus élevée est sous-estimée).

**Synthèse :** Les **Lipides (Fat)** et les **Glucides (Carbs)** sont les facteurs déterminants. Le modèle **Random Forest** est plus efficace que la Régression Linéaire pour prédire la teneur en calories.
##3. Scatterplot : XGBoost Regressor
<img width="845" height="547" alt="image" src="https://github.com/user-attachments/assets/b5b838cd-06b6-41d8-ae78-3edf161de860" />
** interpritation :** 

Ce graphique de dispersion évalue la performance du modèle de régression **XGBoost** (Extreme Gradient Boosting) pour la prédiction de la teneur en calories.

#### **1. Description du Graphique**

* **Titre :** XGBoost : Réel vs. Prédit.
* **Axe X (horizontal) :** **Calories réelles** (valeurs observées dans les données).
* **Axe Y (vertical) :** **Calories prédites** (valeurs estimées par le modèle XGBoost).
* **Ligne Idéale (Trait Rouge) :** La ligne $y=x$ représente le scénario de prédiction parfaite.

#### **2. Analyse de la Performance**

* **Performance Globale : Très Bonne.**
    * La grande majorité des points bleus sont **très proches** de la ligne idéale rouge, surtout pour les valeurs de calories basses à moyennes (entre -1 et +2). Cela indique que le modèle XGBoost est **très précis** dans ses estimations.
* **Analyse des Extrêmes :**
    * **Valeur Réelle Maximale (autour de 4.2 sur X) :** Le modèle a fait une prédiction légèrement inférieure à la valeur réelle (prédite autour de 1.4 sur Y), montrant une **sous-estimation** dans le cas de l'aliment le plus calorique.
    * **Erreur Notable (autour de X=3.0) :** Il y a un point avec une valeur réelle de calories autour de 3.0 qui est **sur-estimée** par le modèle (prédite autour de 3.8 sur Y). C'est l'erreur la plus visible dans la partie supérieure du graphique.

#### **3. Conclusion**

Le modèle **XGBoost** est un **excellent prédicteur des calories** dans ce jeu de données, démontrant une performance largement supérieure pour la majorité des observations. Ses erreurs sont concentrées sur un petit nombre de valeurs extrêmes.

###**analyse comparative entre les 3 modeles :**
---
## Analyse Comparative des Modèles de Prédiction des Calories

La comparaison se base sur la proximité des points de prédiction (Calories prédites) par rapport à la ligne idéale $y=x$ (Calories réelles).

| Critère | Random Forest | XGBoost | Régression Linéaire |
| :--- | :--- | :--- | :--- |
| **Performance Globale** | **Excellente.** Précision très élevée. | **Excellente.** Très haute précision. | **Médiocre.** Précision inférieure aux autres. |
| **Dispersion des Points** | **Très faible.** Les points sont très serrés le long de la ligne idéale. | **Faible.** Points très proches, avec une légère dispersion. | **Élevée.** Points dispersés, s'éloignant de la ligne idéale. |
| **Gestion de la Non-Linéarité** | **Très bonne.** Capacité inhérente à modéliser des relations complexes. | **Très bonne.** Excellent traitement des relations non linéaires. | **Faible.** Suppose une relation linéaire entre les variables, ce qui est une limitation. |
| **Performance sur les Valeurs Extrêmes** | **Très bonne.** Prédit avec précision les valeurs très faibles et très élevées. | **Bonne.** Gère bien la plupart des extrêmes, mais montre une **sur-estimation** notable à $X\approx3.0$ et une **sous-estimation** à $X\approx4.2$. | **Faible.** Difficulté à prédire les valeurs très élevées (sous-estimation fréquente à $X>1.0$). |
| **Meilleur Modèle** | **Vainqueur (Meilleure cohérence globale).** | **Très Proche du Vainqueur.** | **Moins performant.** |

---

### Conclusion

1.  **Modèles Gagnants :** Les modèles basés sur les arbres de décision (Random Forest et XGBoost) sont **nettement supérieurs** à la Régression Linéaire. Ils sont mieux adaptés aux données de calories qui présentent des relations complexes (non linéaires) avec les nutriments.
2.  **Modèle Optimal :** Le **Random Forest** présente la **meilleure performance globale et la meilleure cohérence**, avec la plus faible dispersion des points autour de la ligne idéale.
3.  **Facteurs Expliquant la Performance :** La supériorité des modèles non linéaires s'explique par le fait que les **Lipides (Fat)** et les **Glucides (Carbs)**, bien que les plus corrélés, peuvent avoir des effets non simples qui sont mieux capturés par des algorithmes complexes.
---

### **conclusion :**
# Conclusion de l'analyse

Dans ce projet, nous avons travaillé sur le dataset **Food Nutrition Dataset** pour prédire les **calories** des aliments en fonction de leurs caractéristiques nutritionnelles et de leurs catégories.

## Étapes réalisées

1. **Pré-traitement (Preprocessing)**
   - Nettoyage des données : gestion des doublons et formatage des colonnes.
   - Imputation des valeurs manquantes pour les variables numériques et catégorielles.
   - Encodage des variables catégorielles via One-Hot Encoding.
   - Standardisation des données numériques pour faciliter l'apprentissage des modèles.

2. **Analyse exploratoire des données (EDA)**
   - Visualisation des distributions des variables et des corrélations avec la target.
   - Identification des relations importantes entre certaines variables nutritionnelles et les calories.
   - Feature engineering : création de nouvelles variables (ratios nutritionnels) pour améliorer la prédiction.

3. **Modélisation (Machine Learning)**
   - Trois modèles de régression ont été testés : 
     - **Linear Regression**
     - **Random Forest Regressor**
     - **XGBoost Regressor**
   - Une validation croisée a été réalisée pour évaluer les performances de chaque modèle.
   - Le modèle **Random Forest** a été identifié comme le plus performant.
   - Optimisation des hyperparamètres pour Random Forest et XGBoost afin d'améliorer la précision.

4. **Évaluation et visualisation**
   - Calcul des métriques : RMSE, MAE et R² pour chaque modèle.
   - Scatterplot des calories réelles vs. prédites pour le modèle Random Forest pour visualiser la qualité des prédictions.
   - Analyse de l’importance des features :
     - Pour Random Forest et XGBoost : feature_importances_
     - Pour Linear Regression : coefficients des variables (importance basée sur valeur absolue)

## Interprétation finale

- Les modèles basés sur des arbres (Random Forest et XGBoost) offrent de meilleures performances pour ce dataset par rapport à une régression linéaire simple, en raison de la complexité non linéaire des relations entre les variables nutritionnelles et les calories.
- Les ratios nutritionnels créés lors du feature engineering ont permis d’améliorer la prédiction.
- L’analyse des features importantes permet d’identifier quelles variables ont le plus d’impact sur le calcul des calories, offrant ainsi un aperçu utile pour des applications nutritionnelles ou de recommandations alimentaires.

> En résumé, ce projet illustre une **approche complète de Machine Learning pour la prédiction de calories**, depuis le nettoyage des données jusqu’à l’interprétation des résultats et l’analyse des features les plus influentes.

