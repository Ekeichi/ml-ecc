
# MODEL FOR MARKET PREDICTION

**ANTOINE BOUBEE & MATHEO RUANLT – MACHINE LEARNING**

## I.	Introduction

### a.	La finance

Définition : La finance renvoie à un domaine d'activité, celui du financement , aujourd'hui mondialisé, qui consiste à fournir ou trouver l'argent ou les « produits financiers » nécessaire à la réalisation d'une opération économique. La finance permet de faire transiter des capitaux des agents économiques excédentaires (qui disposent d'une épargne à faire fructifier) aux agents économiques déficitaires, qui en ont besoin (pour se financer, croître, etc.). [1]

La prédiction de l’évolution des marchés est impossible, il est inconcevable de prédire comment le sentiment humain réagis à une nouvelle géopolitique, économique ou encore politique influant sur un marché. Les crises nous prouvent qu’il est difficile de comprendre ou d’interpréter la finance en général. Cependant, malgré des prédictions hasardeuses faites par les humains, il est possible d’avoir de l’intuition quant à la croissance ou la décroissance d’une action ou d’un titre. Par exemple à l’annonce d’une nouvelle (Apple annonce la sortie du vision Pro), nous pouvons avoir l’intuition que le prix de l’action Apple monteras car l’entreprise fera des profits, les actionnaires vont peut-être acheter. Tout cela n’est qu’hypothétique évidement, et difficilement prédictible. Les données sont de plus souvent nombreuses (prix, stock, variation, corrélation entre actions, ETF, nouvelles, politique, …).

Les différents fonds de financement et de trading ont pour seul but d’avoir un PnL positif à la fin de l’année et surtout de battre le marché. Une des principales sources permettant de faire de gros profits est l’information, la donnée sur les entreprises cotées en bourses. Ces fonds payent pour recevoir des données confidentielles sur les entreprises où ils investissent, et arrivent parfois à « battre le marché ». [2]

En tant que particulier, il est simplement impossible d’avoir accès à de telles données, difficile est d’avoir les compétences pour les interpréter et très compliqué d’avoir les financements nécessaires pour réaliser des profits additionnels conséquent. 

### b.	Le Machine Learning 

Depuis des décennies, les modèles de régression de séries temporelles ont constitué une pierre angulaire dans le développement des méthodes d'évaluation financière. Cette approche est essentielle non seulement dans les modèles financiers traditionnels, mais aussi dans l'intelligence artificielle pour la prévision financière, un domaine marqué par la complexité et l'imprévisibilité des tendances du marché.

L'analyse traditionnelle des marchés financiers adopte le modèle à trois facteurs de Fama-French (FFM) et la théorie de l'évaluation par arbitrage (APT) de Chen, Roll et Ross, qui sont toutes deux essentielles dans la détermination des prix des actifs. Ces modèles utilisent la régression linéaire pour analyser les rendements, mais ne se concentrent pas sur les hauts et les bas spécifiques du marché. La dépendance de ces deux modèles aux données historiques limite leur efficacité à anticiper les changements futurs du marché ou les événements sans précédent tels que les crises financières.

Cependant, les techniques émergentes d'apprentissage automatique (ML) se sont révélées prometteuses pour surmonter ces limitations. Des études antérieures démontrent l'efficacité du ML par rapport aux modèles traditionnels. De plus, Billah et Bhuiyan ont souligné la supériorité de l'intégration du prix des actions et du sentiment des nouvelles dans les techniques d'apprentissage profond (DL) pour la prédiction du marché boursier. Ces méthodes émergentes, utilisant des modèles tels que la mémoire à long terme (LSTM), le réseau neuronal récurrent (RNN), et les méthodes d'apprentissage par renforcement (RL), ont démontré une amélioration substantielle dans la prévision des mouvements du marché, un aspect crucial où les modèles traditionnels étaient insuffisants.

La théorie moderne du portefeuille, par Harry Markowitz, souligne la corrélation du marché. Des études récentes ont mis en évidence une forte corrélation positive entre les informations de sentiment, y compris les actualités, les blogs et les médias sociaux, et les tendances du marché boursier. L'avènement de modèles de langage étendus (LLM) avancés tels que ChatGPT et GPT-4 développés par OpenAI a considérablement amélioré la précision de l'analyse des sentiments dans ce contexte. Alors que la recherche de Lopez-Lira mentionnait que les LLM, comme GPT-3, avaient du mal à prévoir avec précision les rendements du marché, les modèles de pointe, comme GPT-4, ont atteint les ratios de Sharpe les plus élevés, démontrant une fiabilité accrue.

Cependant, le manque d'ensembles de données complets et intégrés a considérablement limité l'avancement de la recherche, en particulier dans la mise en œuvre de modèles plus sophistiqués basés sur la technologie des transformateurs, qui pourraient améliorer considérablement l'analyse financière. Pour combler cette lacune, des ensembles de données antérieurs, tels que les nouvelles de Philips provenant de Bloomberg et Reuters, et les nouvelles de Yutkin provenant de Lenta, ainsi que les contributions de sources comme Benzinga, ont été précieux. Néanmoins, ces ensembles de données manquent souvent d'un volume de données suffisant pour l'entraînement de grands modèles et n'incluent pas toujours les prix des actions correspondants. C'est dans ce contexte que l'introduction d'un ensemble de données vaste et diversifié, intégrant à la fois les prix des actions et les informations de sentiment issues des nouvelles financières, comme le Financial News and Stock Price Integration Dataset (FNSPID), devient cruciale pour faire progresser la modélisation et l'analyse prédictives dans le domaine financier. [3]

# II. Developement
 
Sources :
[1] Définition de la finance : https://fr.wikipedia.org/wiki/Finance
[2] Pierre Chartier : X2007, Head of trading vol et hybride, PnL Price maker.
[3] FNSPID : https://arxiv.org/pdf/2402.06698


