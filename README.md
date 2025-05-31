# 📊 Analyse prédictive du cours d'Apple (AAPL) - Machine Learning & Trading algorithmique

## 🎯 Objectif du projet

Ce projet présente une analyse complète et une stratégie de trading algorithmique basée sur l'apprentissage automatique pour prédire les mouvements du cours d'Apple Inc. (AAPL). L'objectif est de développer un modèle prédictif robuste capable de générer des signaux de trading rentables tout en gérant les risques de manière appropriée.

## 📁 Structure du projet

```
projet/
├── aaplAnalysis_wo_sentiment.ipynb    # Notebook principal d'analyse
├── dataset/
│   ├── full_history/AAPL.csv         # Données historiques d'Apple
│   └── googleTrend.csv                # Données de tendances Google
├── README.md                          # Documentation (ce fichier)
└── [autres fichiers auxiliaires]
```

## 🚀 Fonctionnalités principales

- ✅ **Ingénierie de caractéristiques avancée** : autour de 40 indicateurs techniques
- ✅ **Preprocessing robuste** : Transformation et normalisation des données
- ✅ **Validation croisée temporelle** : Respect de l'ordre chronologique
- ✅ **Ensemble de modèles optimisés** : XGBoost, Random Forest, Gradient Boosting, etc.
- ✅ **Stratégie de trading complète** : Avec gestion des risques et coûts de transaction
- ✅ **Interprétabilité des modèles** : Analyse SHAP pour comprendre les prédictions
- ✅ **Métriques financières** : Évaluation orientée trading

## 📖 Description détaillée des cellules

### 🏠 **Cellule 0 : Introduction et méthodologie**
- **Type** : Markdown
- **Contenu** : Présentation du projet, objectifs et améliorations apportées
- **Importance** : Documentation et contexte du projet

### 📊 **Cellule 1 : En-tête - Importation et traitement des données**
- **Type** : Markdown  
- **Contenu** : Titre de la première section principale
- **Rôle** : Structure et organisation du notebook

### 🔧 **Cellule 2 : Chargement et exploration des données**
```python
# Fonctionnalités principales :
- Chargement du fichier AAPL.csv depuis dataset/full_history/
- Conversion et tri chronologique des dates
- Calcul des variations quotidiennes (Pct_Change_Next_Day)
- Visualisation de la distribution des variations
```
**Objectif** : Préparer les données de base et comprendre leur distribution

### 🛠️ **Cellule 3 : Description de l'Ingénierie des Caractéristiques**
- **Type** : Markdown
- **Contenu** : Documentation détaillée des indicateurs techniques utilisés
- **Catégories couvertes** :
  - Indicateurs de tendance (SMA, EMA, MACD, ADX, PSAR)
  - Indicateurs de momentum (RSI, Stochastic, Williams %R, ROC)
  - Indicateurs de volatilité (Bollinger Bands, ATR, volatilité historique)
  - Indicateurs de volume (OBV, VWAP, A/D Line)
  - Features d'interaction et ratios

### ⚙️ **Cellule 4 : Implémentation de l'Ingénierie des Caractéristiques**
```python
# Plus de 40 indicateurs techniques créés :
- Moyennes mobiles : SMA_20, SMA_50, EMA_20, EMA_50
- Momentum : RSI_14, Stochastic, Williams_R, ROC_10
- Volatilité : Bollinger Bands, ATR_14, Hist_Vol_20
- Volume : OBV, VWAP, AD_Line
- Tendance : MACD, ADX, PSAR
- Features d'interaction : RSI_MACD_Interaction, Close_SMA20_Ratio
- Support/Résistance : Pivot Points, Support_1, Resistance_1
```
**Améliorations** : 
- Correction des FutureWarnings liés à `pct_change`
- Ajout de `fill_method=None` pour éviter les avertissements

### 📈 **Cellule 5 : Intégration des Données Google Trends**
```python
# Traitement des données de tendances :
- Lecture des données mensuelles Google Trends
- Interpolation temporelle vers des données journalières
- Lissage des tendances 'apple' et 'iphone'
```
**Objectif** : Enrichir le modèle avec des données de sentiment du marché

### 🔗 **Cellule 6 : Fusion des Datasets**
```python
# Combinaison des données :
- Merge entre données financières et Google Trends
- Tri chronologique final
- Nettoyage des colonnes redondantes
```
**Résultat** : Dataset unifié avec 51 colonnes de features

### 🧹 **Cellule 7-8 : Nettoyage et Preprocessing des Données**
```python
# Pipeline de nettoyage avancé :
- Détection et traitement des valeurs manquantes/infinies
- Imputation par la médiane (SimpleImputer)
- Clipping des valeurs extrêmes (percentiles 1% et 99%)
- Validation de l'intégrité des données
```
**Importance** : Garantit la qualité des données pour l'entraînement

### 🎯 **Cellule 9-10 : Création de la Variable Cible**
```python
# Définition de la variable à prédire :
- Calcul des rendements futurs (1, 3, 5, 10 jours)
- Classification tri-classe : Baisse (0), Neutre (1), Hausse (2)
- Seuil de signification : 0.5% (threshold = 0.005)
- Variable cible principale : prédiction à 5 jours
```
**Amélioration** : Correction de la cible (maintenant vraiment à 5 jours vs 1 jour précédemment)

**Visualisations** :
- Distribution des classes de la variable cible
- Histogramme des rendements à 5 jours
- Analyse d'autocorrélation pour vérifier l'efficience du marché
- Matrice de corrélation entre différentes périodes de prédiction

### 🔄 **Cellule 11-12 : Preprocessing Avancé et Validation Croisée**
```python
# Pipeline de preprocessing sophistiqué :
1. Division temporelle (80/20) - respect de l'ordre chronologique
2. PowerTransformer (Yeo-Johnson) - normalisation des distributions
3. RobustScaler - standardisation robuste aux outliers
4. SelectKBest - sélection des 30 meilleures features par information mutuelle
5. TimeSeriesSplit - validation croisée temporelle (5 folds)
```

**Fonctionnalités clés** :
- Évite le data leakage temporel
- Sélection automatique des features les plus prédictives
- Visualisation des splits de validation croisée
- Graphique d'importance des features sélectionnées

### ⚖️ **Cellule 13-14 : Équilibrage des Classes**
```python
# Traitement du déséquilibre des classes :
- Application de SMOTE (Synthetic Minority Oversampling)
- Équilibrage de la distribution (1789 vs 1789)
- Amélioration de la robustesse du modèle
```

### 🤖 **Cellule 15 : Entraînement et Optimisation des Modèles**
```python
# Ensemble de modèles avec optimisation d'hyperparamètres :
1. XGBoost - Gradient boosting optimisé
2. Random Forest - Forêt d'arbres décisionnels
3. Gradient Boosting - Boosting classique
4. Logistic Regression - Modèle linéaire
5. Neural Network - Réseau de neurones (MLP)
6. Ensemble Pondéré - Combinaison optimale des modèles
```

**Optimisation** :
- RandomizedSearchCV avec 30 itérations par modèle
- Validation croisée temporelle (TimeSeriesSplit)
- Métriques multiples : F1, AUC, Accuracy, Precision, Recall
- Pondération de l'ensemble basée sur les performances CV

**Métriques financières** :
- Profit Score : Métrique orientée trading
- Comparaison CV vs Test pour détecter l'overfitting

### 📊 **Cellule 16 : Aperçu des Données de Test**
- **Type** : Code simple
- **Contenu** : Affichage des premières lignes du dataset de test
- **Utilité** : Vérification des données avant application de la stratégie

### 💹 **Cellule 17 : Stratégie de Trading Algorithmique (CORRIGÉE)**
```python
# Stratégie de trading complète avec gestion des risques :

# Paramètres dynamiques :
- Seuil de confiance : 75e percentile des probabilités du modèle
- Stop-loss : -2% (limitation des pertes)
- Take-profit : +3% (prise de bénéfices)
- Coûts de transaction : 0.1% par trade
- Période de détention max : 10 jours

# Logique de trading :
1. Analyse des probabilités du modèle
2. Entrée en position si probabilité > seuil (75e percentile)
3. Sortie si : stop-loss, take-profit, holding max, ou signal de vente
4. Gestion complète du cash et des actions
```

**Améliorations majeures** :
- **Seuil adaptatif** : Basé sur la distribution réelle des probabilités
- **Gestion des actions** : Calcul précis du nombre d'actions
- **Journal de trading** : Tracking détaillé de tous les trades
- **Métriques de performance** : Sharpe ratio, drawdown, volatilité

**Visualisations** :
- Évolution du portefeuille vs Buy & Hold
- Prix et positions de trading
- Distribution des probabilités avec seuils
- Analyse des trades (rendement vs période de détention)
- Répartition des raisons de sortie

### 🔍 **Cellule 18 : Interprétabilité des Modèles (SHAP)**
```python
# Analyse d'interprétabilité avec SHAP :
- Installation automatique de SHAP si nécessaire
- Analyse du meilleur modèle tree-based (XGBoost)
- Calcul des valeurs SHAP sur un échantillon de 500 observations
- Visualisations multiples :
  * Summary plot (importance globale)
  * Bar plot (importance moyenne)
  * Heatmap des interactions entre features
  * Dependence plots pour les top features
```

**Fonctionnalités d'analyse** :
- Top 10 des features les plus importantes
- Analyse des interactions entre variables
- Compréhension des décisions du modèle
- Alternative avec coefficients de régression logistique

### 📋 **Cellule 19 : Conclusions et Recommandations**
- **Type** : Markdown
- **Contenu** : 
  - Synthèse des performances des modèles
  - Analyse de la stratégie de trading
  - Insights sur l'interprétabilité
  - Recommandations pour le déploiement

### ⭐ **Cellule 20 : Cellule Vide**
- Cellule vide pour extensions futures

## 🛠️ Technologies Utilisées

### **Bibliothèques de Machine Learning**
- `scikit-learn` : Modèles ML, preprocessing, métriques
- `xgboost` : Gradient boosting optimisé
- `imblearn` : Gestion du déséquilibre des classes (SMOTE)
- `shap` : Interprétabilité des modèles

### **Analyse Technique et Financière**
- `ta` (Technical Analysis) : Calcul des indicateurs techniques
- `pandas` : Manipulation des données temporelles
- `numpy` : Calculs numériques optimisés

### **Visualisation**
- `matplotlib` : Graphiques de base
- `seaborn` : Visualisations statistiques avancées

## 📈 Indicateurs Techniques Implémentés

### **Tendance (8 indicateurs)**
- SMA (Simple Moving Average) - 20, 50 périodes
- EMA (Exponential Moving Average) - 20, 50 périodes  
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Parabolic SAR
- Différence entre moyennes mobiles

### **Momentum (6 indicateurs)**
- RSI (Relative Strength Index) - 14 périodes
- Stochastic Oscillator - %K et %D
- Williams %R - 14 périodes
- ROC (Rate of Change) - 10 périodes
- Interactions RSI-MACD et RSI-Stochastic

### **Volatilité (8 indicateurs)**
- Bollinger Bands (haute, moyenne, basse, largeur)
- ATR (Average True Range) - 14 périodes et pourcentage
- Volatilité historique - 20 périodes (annualisée)
- Volatilité glissante - 10 et 30 périodes

### **Volume (3 indicateurs)**
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- A/D Line (Accumulation/Distribution)

### **Support/Résistance (3 indicateurs)**
- Pivot Point
- Support niveau 1
- Résistance niveau 1

### **Features Dérivées (15 indicateurs)**
- Ratios prix/moyennes mobiles
- Ratios prix/volume
- Position relative aux Bollinger Bands
- Variables de lag (prix et volume)
- Retours historiques multiples périodes
- Données Google Trends (apple, iphone)

## 🔧 Améliorations Apportées

### **Corrections Techniques**
1. **FutureWarnings** : Résolution des avertissements `pct_change`
2. **Variable cible** : Correction pour prédiction réelle à 5 jours
3. **Stratégie de trading** : Seuil adaptatif et gestion des positions

### **Preprocessing Avancé**
1. **PowerTransformer** : Normalisation des distributions asymétriques
2. **RobustScaler** : Standardisation robuste aux outliers
3. **Sélection de features** : Information mutuelle (30 meilleures)

### **Validation Robuste**
1. **TimeSeriesSplit** : Validation croisée temporelle (5 folds)
2. **Division chronologique** : Respect strict de l'ordre temporel
3. **Métriques multiples** : Évaluation complète des modèles

### **Stratégie de Trading**
1. **Gestion des risques** : Stop-loss, take-profit automatiques
2. **Coûts réalistes** : Intégration des frais de transaction
3. **Seuil adaptatif** : Basé sur la distribution des probabilités
4. **Métriques financières** : Sharpe ratio, drawdown, profit score

## 📊 Métriques de Performance

### **Métriques de Classification**
- **AUC** (Area Under Curve) : Capacité de discrimination
- **F1-Score** : Équilibre précision/rappel
- **Precision/Recall** : Qualité des prédictions positives
- **Accuracy** : Taux de classification correcte

### **Métriques Financières**
- **Rendement total** : Performance absolue de la stratégie
- **Rendement annualisé** : Performance normalisée
- **Ratio de Sharpe** : Rendement ajusté du risque
- **Volatilité** : Risque de la stratégie
- **Drawdown maximal** : Perte maximale temporaire
- **Profit Score** : Métrique orientée trading

### **Métriques de Trading**
- **Nombre de trades** : Fréquence de trading
- **Taux de réussite** : Pourcentage de trades gagnants
- **Gain/Perte moyens** : Performance moyenne par trade
- **Période de détention** : Durée moyenne des positions

## 🚀 Instructions d'Utilisation

### **Prérequis**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost ta imblearn shap
```

### **Exécution**
1. Placer les fichiers de données dans le dossier `dataset/`
2. Ouvrir le notebook `aaplAnalysis_wo_sentiment.ipynb`
3. Exécuter les cellules dans l'ordre
4. Les résultats s'afficheront automatiquement

### **Structure des Données Requises**
```
dataset/
├── full_history/AAPL.csv (colonnes: date, open, high, low, close, adj_close, volume)
└── googleTrend.csv (colonnes: date, apple, iphone)
```

## 🎯 Résultats Attendus

### **Performance du Modèle**
- AUC > 0.60 (capacité prédictive significative)
- Validation croisée temporelle cohérente
- Sélection automatique des features les plus prédictives

### **Stratégie de Trading**
- Génération de signaux de trading basés sur le ML
- Gestion automatique des risques (stop-loss/take-profit)
- Comparison avec stratégie Buy & Hold
- Métriques de performance détaillées

### **Interprétabilité**
- Identification des facteurs les plus prédictifs
- Analyse des interactions entre variables
- Compréhension des décisions du modèle

## 🔮 Extensions Possibles

### **Améliorations du Modèle**
- Intégration de données alternatives (news, sentiment)
- Modèles deep learning (LSTM, Transformer)
- Feature engineering plus sophistiqué

### **Stratégie de Trading**
- Optimisation des paramètres par backtesting
- Position sizing dynamique
- Diversification multi-actifs

### **Déploiement**
- API de trading en temps réel
- Monitoring automatique des performances
- Alertes et notifications

## 📞 Support et Contributions

Ce projet est développé dans un contexte éducatif. Pour toute question ou suggestion d'amélioration, n'hésitez pas à contribuer ou à signaler des problèmes.

---

**⚠️ Avertissement** : Ce projet est à des fins éducatives uniquement. Les stratégies de trading présentées ne constituent pas des conseils financiers. Investir en bourse comporte des risques de perte en capital.
