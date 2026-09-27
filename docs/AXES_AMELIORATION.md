# Axes d’amélioration de Cs2edge

Audit du 10 novembre 2026, sur le commit `b48231a`, avec Python 3.14.8 et
NumPy 2.5.4. Les constats viennent de la lecture du code, du backtest et de
reproductions locales. Ce document propose les changements à réaliser.

La priorité est de rendre l’évaluation représentative du comportement réellement
servi et de garantir l’intégrité des données et du journal.

| Ordre | Priorité | Axe | Impact | Effort estimé |
| --- | --- | --- | --- | --- |
| 1 | P1 | Aligner prédiction et backtests | Les scores doivent mesurer les probabilités effectivement annoncées | Faible à moyen |
| 2 | P1 | Isoler la validation du réglage | Le gain publié doit porter sur des matchs exclus du choix des paramètres | Moyen |
| 3 | P1 | Garantir la date de connaissance des informations | Éviter une prédiction historique alimentée par des résultats futurs et détecter les données anciennes | Moyen |
| 4 | P1 | Identifier les matchs dans le journal | Éviter un règlement sur le mauvais résultat et les doublons | Moyen |
| 5 | P1 | Valider et préserver les données importées | Empêcher des exemples erronés d’entrer dans les ratings et l’apprentissage | Moyen |
| 6 | P2 | Corriger les handicaps BO5 du catalogue | Associer la bonne probabilité à chaque ligne | Faible |
| 7 | P2 | Corriger la portée des options CLI | Utiliser le fichier et les options demandés, quelle que soit leur position | Faible |
| 8 | P2 | Renforcer les données et la validation du modèle | Mesurer la robustesse hors du seul générateur actuel | Moyen à élevé |

**1. Aligner prédiction et backtests**

Dans [predict.py](../cs2edge/predict.py#L321), `Analyzer.predict` appelle
`combine_probability` sans `certainty`. Dans
[backtest.py](../cs2edge/backtest.py#L153), `walk_forward` fournit cette valeur.
Le facteur qui réduit la confiance suit donc deux règles différentes.
`simulate_tipster` omet également `certainty` dans son appel, vers la ligne 729
de [pronostic.py](../cs2edge/pronostic.py#L729).

Reproduction sur les 496 paires des 32 équipes du CSV fourni, au 6 septembre
2026, avec les mêmes modèles entraînés et les mêmes probabilités brutes :
179 paires changent de plus d’un point de probabilité lorsque l’on applique le
garde-fou du backtest. Pour `gamerlegion` contre `passion`, la probabilité de A
passe de **5,62 % à 10,20 %**, soit **4,58 points**. Cette comparaison isole
le garde-fou ; elle ne compare pas des modèles entraînés sur des périodes différentes.

Un deuxième écart concerne le GBDT : `Analyzer.train(use_gbdt=True)` conserve
`True` pour le modèle de maps, car `map_kwargs.setdefault("use_gbdt", False)`
n’écrase pas la valeur transmise. Les deux backtests imposent `False` à ce
modèle. La configuration diverge donc lorsque l’option est activée ; son effet
numérique reste à vérifier avec scikit-learn installé.

Changement proposé : partager une configuration d’inférence et un calcul du
contexte de fiabilité entre les trois parcours. Décider explicitement si le
modèle de maps utilise le GBDT et appliquer ce choix partout.

Critère de réussite : à modèles, features, date et configuration identiques,
la prédiction unitaire et les deux relectures donnent la même probabilité,
y compris pour une équipe peu connue ou inactive et avec l’option GBDT.

**2. Isoler la validation du réglage**

Dans [backtest.py](../cs2edge/backtest.py#L269), `tune` affecte
`fit_matches, holdout = matches[:cut], matches`. Le score présenté comme
validation est donc un backtest sur l’historique complet.

En traçant les identifiants effectivement évalués pendant un appel à `tune`
avec `holdout_frac=0.25` et une grille réduite à `k_base=12` :

- La période réservée devrait commencer le **6 décembre 2025**.
- Le score de validation commence le **20 décembre 2024**.
- **772 des 1 376 matchs évalués, soit 56,1 %, appartiennent à la période de
  réglage**.

La période finale est bien exclue de la recherche initiale, mais le score final
mélange ses résultats avec ceux de la période utilisée pour choisir les paramètres.

Changement proposé : conserver tout le passé nécessaire au calcul des ratings
et à l’entraînement, puis calculer les métriques de validation uniquement sur
les identifiants de la période réservée. Découper simplement l’entrée en
`matches[cut:]` ferait perdre l’historique utile. Le poids `best_blend`, choisi
sur les résultats évalués, doit aussi rester exploratoire jusqu’à une validation
ultérieure si l’on souhaite l’adopter.

Critère de réussite : l’intersection entre les identifiants de réglage et ceux
du score de validation est vide ; modifier les résultats de la période réservée
ne change pas la configuration retenue pendant le réglage.

**3. Garantir la date de connaissance des informations**

Dans [predict.py](../cs2edge/predict.py#L276), changer `on` ne reconstruit ni
l’état des équipes ni l’entraînement. Un appel demandant `navi` contre `g2`
au **1er janvier 2025** utilise un modèle dont `last_date` vaut le
**6 septembre 2026** et affiche des derniers résultats d’août/septembre 2026.

La date implicite est également celle du dernier résultat disponible. Sur le
CSV fourni, une prédiction sans date annonce le 6 septembre 2026, une confiance
« correcte » et aucune réserve. En demandant explicitement le 10 novembre 2026,
la même affiche devient de confiance « faible », avec 84 jours d’inactivité
pour l’équipe la plus ancienne. La probabilité de NAVI passe de 93,51 % à 85,36 %.

Enfin, les dates sont stockées au jour et départagées par `match_id`. Quatre des
dix frontières du backtest fourni coupent une même journée. Sans heure de
disponibilité des résultats, ce protocole ne garantit pas que toutes les
informations utilisées étaient connues au moment de chaque prédiction.

Changement proposé : définir une date de connaissance explicite. Pour l’usage
courant, utiliser la date demandée ou la date du jour ; rejeter une demande
antérieure aux informations chargées, ou reconstruire et réentraîner sur le
passé autorisé. Utiliser des horodatages de disponibilité lorsque la source
les fournit. À défaut, calculer les prédictions d’une journée avant d’intégrer
ses résultats et aligner les frontières des folds sur les journées.

Critère de réussite : ajouter des résultats postérieurs ne change pas une
prédiction historique reconstruite ; l’ancienneté est calculée à la date réelle
de la demande ; aucune information indisponible n’entre dans un fold.
