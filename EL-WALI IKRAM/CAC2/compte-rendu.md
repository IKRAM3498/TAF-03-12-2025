\documentclass[11pt, a4paper]{article}

% --- UNIVERSAL PREAMBLE BLOCK ---
\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2cm, right=2cm]{geometry}
\usepackage{fontspec}

\usepackage[french, bidi=basic, provide=*]{babel}

\babelprovide[import, onchar=ids fonts]{french}
\babelprovide[import, onchar=ids fonts]{english}

% Set default/Latin font to Sans Serif in the main (rm) slot
\babelfont{rm}{Noto Sans}
% Assign a specific font for French text (optional, but good practice for consistency)
\babelfont[french]{rm}{Noto Sans}

\usepackage{enumitem}
\setlist[itemize]{label=-}
% --- END UNIVERSAL PREAMBLE BLOCK ---

% Packages spécifiques pour un rapport scientifique
\usepackage{amsmath}    % Pour les maths (RMSE, R2)
\usepackage{amssymb}
\usepackage{graphicx}   % Pour l'inclusion d'images/graphiques (nécessaire si vous ajoutez des figures)
\usepackage{booktabs}   % Pour de meilleurs tableaux
\usepackage{caption}    % Pour les légendes de figures/tableaux
\usepackage{subcaption} % Pour les sous-figures (si besoin)
\usepackage{listings}   % Pour l'inclusion de code (si besoin)
\usepackage{float}      % Pour forcer le positionnement des figures
\usepackage{hyperref}   % Pour les liens cliquables (doit être le dernier)

% Commandes personnalisées
\newcommand{\todo}[1]{\textcolor{red}{\textbf{[À FAIRE : #1]}}} % Commande pour les tâches à compléter
\newcommand{\titre}{Rapport Scientifique: Modélisation et Prédiction des Calories Alimentaires}
\newcommand{\auteur}{EL WALI IKRAM}
\newcommand{\dateRapport}{\today}
\newcommand{\modeleMeilleur}{Random Forest} % À remplacer par votre meilleur modèle

% Configuration des titres
\title{\titre}
\author{\auteur}
\date{\dateRapport}

% Pour la couleur des liens hyperréférences
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan,
    pdftitle={\titre},
    pdfauthor={\auteur},
}

\begin{document}

\maketitle
\newpage
\tableofcontents
\newpage

% --------------------------------------------------------------------------------------------------
% 1. INTRODUCTION (Page 1-2)
% --------------------------------------------------------------------------------------------------
\section{Introduction}

\subsection{Contexte et Problématique}
La nutrition et la gestion des apports caloriques sont au cœur des préoccupations de santé publique et du bien-être individuel. Avec la prolifération des données sur la composition des aliments, issues de bases de données gouvernementales ou d'applications mobiles, le défi réside dans la capacité à extraire de la valeur de ces informations brutes. L'estimation précise des calories (énergie métabolisable) est cruciale, non seulement pour les consommateurs, mais aussi pour les professionnels de la nutrition et l'industrie agroalimentaire.

La méthode traditionnelle de calcul des calories, basée sur les coefficients d'Atwater (4 kcal/g pour les protéines et les glucides, 9 kcal/g pour les lipides), est une approximation. Elle ne prend pas en compte la complexité des interactions nutritionnelles ni la biodisponibilité réelle des macronutriments. Dès lors, l'application des techniques de Machine Learning (ML) se présente comme une approche prometteuse pour modéliser cette relation de manière plus nuancée, en exploitant un ensemble plus large de caractéristiques nutritionnelles.

\subsection{Objectifs du Projet}
Ce projet de Machine Learning s'articule autour de trois objectifs principaux :
\begin{enumerate}
    \item \textbf{Exploration et Préparation des Données :} Mener une analyse exploratoire approfondie du jeu de données \textit{Food Nutrition Dataset} et le préparer pour la modélisation en gérant les valeurs manquantes, en standardisant les échelles et en réalisant du \textit{Feature Engineering}.
    \item \textbf{Modélisation Prédictive :} Développer et comparer plusieurs modèles de régression (Régression Linéaire, Random Forest, XGBoost) capables de prédire la quantité de calories (\textit{Calories}) d'un aliment en fonction de sa composition nutritionnelle (protéines, lipides, glucides, etc.).
    \item \textbf{Évaluation et Interprétation :} Évaluer les performances des modèles à l'aide de métriques de régression pertinentes (RMSE, MAE, $R^2$) et analyser l'importance des variables pour déterminer quels nutriments influencent le plus la valeur calorique.
\end{enumerate}

\subsection{Structure du Rapport}
Le présent rapport est structuré conformément aux étapes d'un projet de Machine Learning. La Section \ref{sec:methodologie} détaillera les choix techniques effectués pour le nettoyage, la sélection et l'ingénierie des variables. La Section \ref{sec:resultats} présentera les résultats comparés des modèles et proposera une analyse approfondie des performances et des interprétations. Enfin, la Section \ref{sec:conclusion} synthétisera les conclusions du travail et ouvrira sur les perspectives d'amélioration.
\newpage

% --------------------------------------------------------------------------------------------------
% 2. MÉTHODOLOGIE (Page 3-5)
% --------------------------------------------------------------------------------------------------
\section{Méthodologie}
\label{sec:methodologie}

\subsection{Description et Préparation du Jeu de Données}

\subsubsection{Source et Contenu}
Le jeu de données utilisé est le \textit{Food Nutrition Dataset (150+ Everyday Foods)}, qui compile les valeurs nutritionnelles de plus de 150 aliments courants. Chaque observation (ligne) représente un aliment et est caractérisée par des attributs tels que \textit{Protein}, \textit{Carbs}, \textit{Fat}, \textit{Saturated Fat}, \textit{Fiber}, \textit{Sugar}, \textit{Cholesterol}, \textit{Sodium}, et la variable cible, \textit{Calories}.

\subsubsection{Nettoyage des Données (\textit{Data Cleaning})}
La qualité des données est primordiale. Les étapes de nettoyage ont été cruciales :
\begin{enumerate}
    \item \textbf{Gestion des Valeurs Manquantes :} L'analyse exploratoire a révélé des valeurs manquantes dans certaines colonnes. \todo{Spécifier la méthode utilisée pour gérer les NaNs (ex: Imputation par la médiane ou la moyenne, ou suppression des lignes/colonnes).} Le choix s'est porté sur \todo{Imputation/Suppression} pour préserver l'intégrité de l'ensemble de données tout en garantissant la complétude des observations utilisées pour l'entraînement.
    \item \textbf{Gestion des Aberrations (\textit{Outliers}) :} Bien que les valeurs nutritionnelles soient généralement bornées, une vérification a été effectuée pour s'assurer qu'aucune valeur n'était physiquement impossible (ex : une valeur de graisse négative). \todo{Décrire si des outliers ont été détectés et comment ils ont été traités (ex: Winsorisation, suppression ou conservation).}
\end{enumerate}

\subsubsection{Ingénierie des Variables (\textit{Feature Engineering})}
Le \textit{Feature Engineering} est l'étape où de nouvelles variables sont créées pour améliorer la puissance prédictive des modèles.
\begin{itemize}
    \item \textbf{Ratios de Macronutriments :} Pour capturer la densité nutritionnelle relative de l'aliment, des ratios ont été calculés. Les plus pertinents sont :
    \begin{itemize}
        \item \textbf{Ratio Protéines/Lipides :} $\frac{\text{Protein}}{\text{Fat}}$
        \item \textbf{Ratio Fibres/Glucides :} $\frac{\text{Fiber}}{\text{Carbs}}$ (indicateur de la qualité des glucides)
        \item \textbf{Densité Nutritionnelle Globale :} $\frac{\text{Protein} + \text{Carbs} + \text{Fat}}{\text{Total Mass}}$ (en l'absence d'une colonne de masse totale, nous avons utilisé \todo{Préciser si vous avez créé une variable 'Total Macronutrients' ou si vous avez utilisé les ratios directement}).
    \end{itemize}
    \item \todo{Ajouter toute autre variable créée, par exemple la somme des macronutriments, si c'est le cas.} Ces variables dérivées permettent aux modèles de saisir des relations non linéaires ou des proportions qui seraient invisibles aux caractéristiques brutes.
\end{itemize}

\subsection{Sélection et Justification des Modèles}

L'objectif étant la prédiction d'une valeur numérique continue (\textit{Calories}), le problème relève de la \textbf{Régression}. Trois modèles ont été sélectionnés pour leur complémentarité.

\subsubsection{Régression Linéaire (RL)}
\begin{itemize}
    \item \textbf{Justification du Choix :} C'est un modèle de base, simple et interprétable. Il sert de \textbf{référence (\textit{baseline})} pour mesurer la performance des modèles plus complexes. Il suppose une relation linéaire entre les macronutriments et les calories, une hypothèse qui est théoriquement pertinente (coefficients d'Atwater) mais potentiellement simpliste en pratique.
    \item \textbf{Limites Anticipées :} La RL est peu performante si les relations sont non linéaires ou s'il y a de fortes interactions entre les variables (ce qui est souvent le cas en nutrition).
\end{itemize}

\subsubsection{Random Forest (RF)}
\begin{itemize}
    \item \textbf{Justification du Choix :} Le Random Forest est un modèle d'ensemble basé sur l'agrégation de plusieurs arbres de décision. Il est reconnu pour sa robustesse, sa capacité à gérer les relations non linéaires et les interactions complexes entre les variables sans nécessiter de mise à l'échelle des données.
    \item \textbf{Avantage Majeur :} Il fournit une estimation fiable de l'importance des variables (\textit{Feature Importance}), ce qui est essentiel pour l'interprétation nutritionnelle de notre problème.
\end{itemize}

\subsubsection{XGBoost (\textit{eXtreme Gradient Boosting})}
\begin{itemize}
    \item \textbf{Justification du Choix :} XGBoost est un autre algorithme d'ensemble, basé sur l'approche du \textit{Gradient Boosting}. Il est souvent considéré comme l'un des modèles les plus performants dans les compétitions de Machine Learning (Kaggle), notamment pour les données tabulaires. Il construit des arbres de manière séquentielle, chaque nouvel arbre tentant de corriger les erreurs des précédents.
    \item \textbf{Objectif :} Il est utilisé pour évaluer si une complexité accrue par rapport au Random Forest permet d'obtenir une amélioration significative de la performance.
\end{itemize}

\subsection{Protocole d'Évaluation}
\subsubsection{Séparation des Données}
Le jeu de données a été divisé en un ensemble d'entraînement (\textit{Training Set}) et un ensemble de test (\textit{Test Set}) selon une proportion de \todo{Spécifier la proportion, ex: 80\% / 20\%}. L'ensemble de test, non vu par les modèles durant l'apprentissage, est utilisé uniquement pour l'évaluation finale des performances généralisées.

\subsubsection{Métriques de Régression}
Pour évaluer les modèles, les métriques suivantes ont été choisies :
\begin{itemize}
    \item \textbf{Erreur Quadratique Moyenne (\textit{Root Mean Squared Error}, RMSE) :} La métrique principale. Elle mesure l'écart type des résidus (erreurs de prédiction). Elle pénalise fortement les grandes erreurs, ce qui est souhaitable pour la prédiction des calories.
    $$ \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2} $$
    \item \textbf{Erreur Absolue Moyenne (\textit{Mean Absolute Error}, MAE) :} Elle représente la grandeur moyenne des erreurs. Elle est plus facile à interpréter que la RMSE car elle n'utilise pas le carré des erreurs.
    $$ \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i| $$
    \item \textbf{Coefficient de Détermination ($R^2$ Score) :} Il représente la proportion de la variance de la variable dépendante qui est expliquée par les variables indépendantes du modèle. Un $R^2$ proche de 1 indique un ajustement parfait.
\end{itemize}
\newpage

% --------------------------------------------------------------------------------------------------
% 3. RÉSULTATS & DISCUSSION (Page 6-15)
% --------------------------------------------------------------------------------------------------
\section{Résultats et Discussion}
\label{sec:resultats}

\subsection{Analyse de l'Exploration des Données (\textit{EDA})}
\todo{Analyser les graphiques de l'EDA et commenter les observations clés de votre notebook.}

\subsubsection{Distribution de la Variable Cible (\textit{Calories})}
L'étude de la distribution de la variable cible est fondamentale. \todo{Décrire la distribution (par exemple, "elle est légèrement asymétrique à droite, indiquant une majorité d'aliments à faible teneur calorique avec quelques outliers à haute teneur").}

\begin{figure}[H]
  \centering
  \framebox{\parbox{0.8\textwidth}{\centering
    \vspace{3cm}
    \textbf{Image Placeholder} \\
    \small\textit{Figure 3.1: Histogramme de la distribution de la variable cible 'Calories'}
    \vspace{3cm}
  }}
  \caption{Visualisation de la distribution des calories dans le jeu de données.}
  \label{fig:hist_calories}
\end{figure}

\subsubsection{Corrélations des Macronutriments}
L'analyse des corrélations entre les variables nutritionnelles est essentielle. La matrice de corrélation a permis de confirmer la relation forte entre les macronutriments (Glucides, Protéines, Lipides) et les Calories. \todo{Décrire la corrélation la plus forte (probablement avec les Lipides, étant donné leur coefficient Atwater de 9 kcal/g).}

\begin{figure}[H]
  \centering
  \framebox{\parbox{0.8\textwidth}{\centering
    \vspace{3cm}
    \textbf{Image Placeholder} \\
    \small\textit{Figure 3.2: Matrice de corrélation des variables nutritionnelles}
    \vspace{3cm}
  }}
  \caption{Matrice de corrélation affichant les dépendances linéaires entre les features.}
  \label{fig:corr_matrix}
\end{figure}
\todo{Analyse des corrélations : Que nous apprend la matrice sur les relations entre les nutriments avant la modélisation ?}

\subsection{Comparaison des Performances des Modèles}

Les trois modèles ont été entraînés sur l'ensemble d'entraînement et évalués sur l'ensemble de test pour déterminer leur capacité de généralisation.

\begin{table}[H]
    \centering
    \caption{Synthèse des Métriques de Performance sur l'Ensemble de Test}
    \label{tab:metriques}
    \begin{tabular}{|l|c|c|c|}
        \toprule
        \textbf{Modèle} & \textbf{RMSE} & \textbf{MAE} & \textbf{\(R^2\) Score} \\
        \midrule
        Régression Linéaire (RL) & \todo{Valeur RMSE RL} & \todo{Valeur MAE RL} & \todo{Valeur R2 RL} \\
        Random Forest (RF) & \todo{Valeur RMSE RF} & \todo{Valeur MAE RF} & \todo{Valeur R2 RF} \\
        XGBoost & \todo{Valeur RMSE XGBoost} & \todo{Valeur MAE XGBoost} & \todo{Valeur R2 XGBoost} \\
        \bottomrule
    \end{tabular}
\end{table}

\subsubsection{Analyse des Métriques}

\paragraph{Régression Linéaire :} Le modèle RL, bien que servant de référence, a obtenu le $R^2$ le plus faible et la RMSE la plus élevée. Cela confirme que la relation entre les nutriments et les calories n'est pas strictement linéaire ou que les interactions entre les features ne sont pas bien capturées par ce modèle simple. Son $R^2$ de \todo{Valeur R2 RL} indique que \todo{Valeur R2 RL * 100}\% de la variance calorique est expliquée, ce qui est acceptable pour un modèle de base mais insuffisant pour une application précise.

\paragraph{Random Forest et XGBoost :} Les modèles basés sur les arbres, RF et XGBoost, ont systématiquement surpassé la Régression Linéaire.
\begin{itemize}
    \item Le modèle \textbf{\modeleMeilleur} a affiché la meilleure performance, avec une RMSE de \todo{Valeur RMSE Modèle Meilleur} et un $R^2$ de \todo{Valeur R2 Modèle Meilleur}. Un $R^2$ si proche de 1 démontre une capacité prédictive exceptionnelle, suggérant que le modèle a réussi à identifier les relations non linéaires complexes et les interactions fines au sein du jeu de données.
    \item L'amélioration observée par rapport à la Régression Linéaire valide l'approche de la modélisation non linéaire pour ce type de données. \todo{Comparer RF et XGBoost : Lequel est le meilleur et pourquoi ? Ex: Si RF est meilleur, cela signifie que la complexité supplémentaire d'XGBoost n'était pas justifiée, ou qu'il a subi un léger surapprentissage.}
\end{itemize}

\subsubsection{Analyse des Erreurs du Meilleur Modèle (\modeleMeilleur)}
L'analyse des résidus du modèle \modeleMeilleur\ est cruciale pour comprendre les limites de ses prédictions.

\begin{figure}[H]
  \centering
  \framebox{\parbox{0.8\textwidth}{\centering
    \vspace{3cm}
    \textbf{Image Placeholder} \\
    \small\textit{Figure 3.3: Scatterplot des calories réelles vs. prédites par le modèle \modeleMeilleur}
    \vspace{3cm}
  }}
  \caption{Nuage de points comparant les valeurs de calories réelles (\textit{y\_true}) et les prédictions du modèle \modeleMeilleur\ (\textit{y\_pred}).}
  \label{fig:scatter_pred}
\end{figure}

\todo{Analyse du Scatterplot :}
\begin{itemize}
    \item \textbf{Idéal vs. Réel :} Idéalement, les points devraient s'aligner le long de la droite d'identité ($y_{pred} = y_{true}$). \todo{Décrire comment les points s'alignent : sont-ils serrés autour de la ligne ?}
    \item \textbf{Erreurs Extrêmes :} \todo{Identifier où se situent les plus grandes erreurs. Par exemple, "Le modèle semble avoir du mal à prédire avec précision les aliments à très haute teneur calorique (au-delà de X kcal) ou ceux à très faible teneur calorique."}
    \item \textbf{Biais :} \todo{Y a-t-il un biais dans la prédiction ? Le modèle sous-estime-t-il (\textit{underestimates}) ou surestime-t-il (\textit{overestimates}) les calories de manière systématique pour certains niveaux ?}
\end{itemize}

\subsection{Interprétabilité et Importance des Variables (\textit{Feature Importance})}

L'un des principaux avantages des modèles arborescents est la capacité à quantifier l'impact de chaque variable d'entrée sur la prédiction finale. Cette analyse est d'une importance capitale dans un contexte nutritionnel.

\begin{figure}[H]
  \centering
  \framebox{\parbox{0.8\textwidth}{\centering
    \vspace{3cm}
    \textbf{Image Placeholder} \\
    \small\textit{Figure 3.4: Importance des features pour le modèle \modeleMeilleur}
    \vspace{3cm}
  }}
  \caption{Bar chart illustrant l'importance relative des variables d'entrée dans le processus de prédiction du modèle \modeleMeilleur.}
  \label{fig:feature_importance}
\end{figure}

\todo{Analyser le graphique d'importance des features et commenter les 4-5 variables les plus importantes. Ce commentaire doit être le cœur de votre discussion.}

\subsubsection{Variables Primordiales}
\begin{itemize}
    \item \textbf{Lipides (\textit{Fat})} : Sans surprise, la variable \textit{Fat} est la plus influente. Sa dominance s'explique par son facteur énergétique (9 kcal/g) qui est plus de deux fois supérieur à celui des autres macronutriments, la rendant l'unique plus grand contributeur à la variance calorique totale.
    \item \textbf{Glucides (\textit{Carbs}) / Protéines (\textit{Protein})} : Ces variables se positionnent également en tête. Il est crucial de noter si l'un est plus important que l'autre dans votre modèle. \todo{Comparer l'importance relative de Carbs vs. Protein selon votre graphique.}
    \item \textbf{Variables Issus du \textit{Feature Engineering}} : L'importance des ratios créés est un indicateur de la pertinence de l'étape de \textit{Feature Engineering}. Si le \textit{Ratio Protéines/Lipides} ou le \textit{Ratio Fibres/Glucides} apparaissent dans le top 10 des features, cela signifie que la \textbf{proportion} des nutriments est un meilleur prédicteur que leur valeur absolue seule, ajoutant une dimension qualitative à la modélisation.
\end{itemize}

\subsubsection{Impact des Autres Nutriments}
Les micronutriments et d'autres variables comme \textit{Sodium}, \textit{Cholesterol} ou \textit{Saturated Fat} montrent généralement une importance bien moindre. \todo{Discuter de l'importance de ces variables.} Si elles ont un impact, il est plus probable qu'elles agissent comme des indicateurs indirects de la catégorie d'aliment (ex : Teneur élevée en sel $\implies$ Aliment transformé) plutôt que des contributeurs directs à l'énergie.

\subsection{Synthèse des Graphiques et Conclusions Intermédiaires}
\todo{Ici, vous devez intégrer une analyse détaillée de tous les autres graphiques que vous avez générés dans votre notebook (ex: Boxplots, autres visualisations spécifiques).}

\begin{enumerate}
    \item \textbf{Graphique : \todo{Nom du graphique 1}}
    \begin{itemize}
        \item \textbf{Description :} \todo{Décrire ce que le graphique montre (ex: Boxplot des calories par catégorie d'aliment).}
        \item \textbf{Analyse :} \todo{Qu'avez-vous appris de ce graphique ? (ex: "Les catégories de viandes et de noix montrent une variabilité calorique plus élevée que les légumes.")}
    \end{itemize}
    \item \textbf{Graphique : \todo{Nom du graphique 2}}
    \begin{itemize}
        \item \textbf{Description :} \todo{Décrire ce que le graphique montre (ex: Distribution du sucre vs. fibres).}
        \item \textbf{Analyse :} \todo{Qu'avez-vous appris de ce graphique ? (ex: "Il y a une corrélation inverse faible entre le sucre et la fibre dans cet échantillon, ce qui est attendu.")}
    \end{itemize}
    \item \textbf{Graphique : \todo{Nom du graphique 3}}
    \begin{itemize}
        \item \textbf{Description :} \todo{Décrire ce que le graphique montre (ex: Residual Plot pour le modèle Random Forest).}
        \item \textbf{Analyse :} \todo{Qu'avez-vous appris de ce graphique ? (ex: "Les résidus sont répartis de manière aléatoire autour de zéro, indiquant une bonne homoscédasticité, sauf pour les prédictions extrêmes où un pattern est visible.")}
    \end{itemize}
    \item \textbf{Graphique : \todo{Nom du graphique 4}}
    \begin{itemize}
        \item \textbf{Description :} \todo{Décrire ce que le graphique montre (ex: Comparaison des coefficients de la Régression Linéaire).}
        \item \textbf{Analyse :} \todo{Qu'avez-vous appris de ce graphique ? (ex: "Les coefficients de la RL sont proches des facteurs d'Atwater (4, 4, 9), confirmant la validité théorique de base de notre dataset.")}
    \end{itemize}
\end{enumerate}
\newpage

% --------------------------------------------------------------------------------------------------
% 4. CONCLUSION (Page 16+)
% --------------------------------------------------------------------------------------------------
\section{Conclusion}
\label{sec:conclusion}

\subsection{Synthèse des Résultats}
Ce projet a démontré l'efficacité des modèles de Machine Learning, en particulier les méthodes d'ensemble comme le \modeleMeilleur, pour la prédiction précise des calories des aliments à partir de leurs profils nutritionnels. Avec un \(R^2\) de \todo{Valeur R2 Modèle Meilleur} sur l'ensemble de test, le modèle dépasse largement la performance de la Régression Linéaire, confirmant que la relation entre les nutriments et les calories est mieux modélisée par des approches non linéaires qui capturent les interactions complexes.

L'analyse de l'importance des variables a corroboré les fondements de la nutrition en plaçant les lipides en tête des prédicteurs, suivi des glucides et des protéines. L'étape d'ingénierie des variables, incluant la création de ratios, a également permis de renforcer la robustesse et l'interprétabilité du modèle.

\subsection{Limites du Modèle Actuel}
Malgré l'excellente performance, le modèle présente des limites qui doivent être reconnues :
\begin{enumerate}
    \item \textbf{Taille et Représentativité du Dataset :} Le jeu de données ne contient qu'environ 150 aliments. Bien qu'il soit suffisant pour une démonstration, un modèle de production nécessiterait un ensemble de données beaucoup plus vaste et diversifié, incluant des aliments composites et transformés, pour garantir une généralisation fiable.
    \item \textbf{Absence de Facteurs de Traitement :} Le modèle ne tient pas compte de l'impact du traitement des aliments, qui peut modifier la biodisponibilité et le taux de calories absorbées par le corps (ex : fibres solubles vs. insolubles, index glycémique).
    \item \textbf{Surapprentissage Potentiel :} Un $R^2$ très élevé (\todo{Valeur R2 Modèle Meilleur}) sur un ensemble de données de petite taille peut soulever des doutes quant au surapprentissage. Une validation croisée (\textit{Cross-Validation}) plus rigoureuse ou l'utilisation de données externes serait nécessaire pour confirmer la robustesse du modèle.
\end{enumerate}

\subsection{Pistes d'Amélioration Futures}
Pour renforcer la précision et l'utilité de ce modèle prédictif, plusieurs axes d'amélioration peuvent être explorés :
\begin{enumerate}
    \item \textbf{Collecte de Données Spécifiques :} Intégrer des données supplémentaires sur des attributs non inclus, tels que la catégorie d'aliment (légume, viande, fruit, produit laitier, etc.) et des variables de classification plus fines (ex: présence d'édulcorants, type de fibres).
    \item \textbf{Optimisation des Hyperparamètres :} Une recherche plus poussée des hyperparamètres (via Grid Search ou Random Search) du modèle \modeleMeilleur\ permettrait d'affiner encore les performances.
    \item \textbf{Modélisation d'Ensemble (\textit{Stacking/Blending}) :} Combiner les prédictions du Random Forest et de l'XGBoost dans un modèle d'ensemble final pourrait potentiellement lisser les erreurs et améliorer la généralisation par rapport à un modèle unique.
    \item \textbf{Détection d'Anomalies :} Mettre en place une phase de détection d'anomalies pour identifier les aliments dont la prédiction calorique est trop éloignée de la réalité (résidus extrêmes), afin de corriger les erreurs de données ou d'identifier des cas spéciaux.
\end{enumerate}

Ce travail jette les bases d'une approche de \textit{data science} appliquée à la nutrition, démontrant que les outils d'apprentissage automatique peuvent fournir des estimations caloriques plus précises et des insights précieux sur l'impact énergétique des différents nutriments.

\end{document}
