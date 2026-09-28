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

**4. Identifier les matchs dans le journal**

Dans [journal.py](../cs2edge/journal.py#L94), l’index de règlement utilise
seulement `(date, paire d’équipes)`. Deux rencontres entre les mêmes équipes
le même jour s’écrasent. Le jeu fourni contient déjà 19 clés de ce type avec
plusieurs rencontres.

Reproduction minimale : deux BO1 `a` contre `b`, le même jour, dans deux
événements différents. A gagne le premier et perd le second. Le pronostic
enregistré pour le premier événement est réglé **« perdu », score 0-1**, car
le second résultat a remplacé le premier dans l’index. Le champ `event` du
journal n’intervient pas dans le rapprochement.

Autre reproduction : `record([pick, pick], path)` ajoute **deux lignes** à un
journal vide. L’ensemble `known` n’est pas mis à jour pendant l’ajout. L’identité
du pronostic ne contient pas non plus d’identifiant de rencontre.

Changement proposé : propager un identifiant stable de rencontre depuis les
fixtures jusqu’au règlement. Pour les anciennes lignes sans identifiant,
conserver les candidats et laisser les cas ambigus en attente. Dédupliquer
aussi à l’intérieur d’un même lot. Ajouter l’heure d’émission, la date limite
des données et la version du modèle pour rendre chaque annonce vérifiable.
Une transaction SQLite ou une écriture atomique avec verrouillage renforcerait
ensuite la résistance aux interruptions et aux écritures concurrentes.

Critère de réussite : deux rencontres le même jour restent distinctes ; changer
l’ordre des résultats ne change pas le règlement ; un doublon dans le même lot
crée une seule ligne ; une ambiguïté ne produit aucun règlement automatique.

**5. Valider et préserver les données importées**

Dans [schema.py](../cs2edge/schema.py#L167), l’import vérifie les colonnes et
convertit les valeurs, mais il accepte par exemple `rounds_a=-2`,
`rounds_b=13.9` et `best_of=2`. Le score décimal est silencieusement tronqué à
13. Une série BO3 contenant seulement une victoire de map obtient déjà un
vainqueur via `Match.a_won`, sans statut indiquant qu’elle est incomplète.

Deux pertes ou divergences de données sont également reproduites :

- `write_csv` puis `read_csv` transforme un roster de cinq joueurs en tuple
  vide : les colonnes de roster ne sont pas écrites.
- `sources.from_records` transforme la chaîne `lan="false"` en `True`, car
  l’import API utilise `bool(...)`, contrairement à l’import CSV.

Changement proposé : partager la normalisation et les validations entre CSV,
API et construction des séries. Vérifier les scores entiers, les formats
supportés, la cohérence des équipes et des indices de map, les doublons et
le statut final des séries. Prévoir explicitement les forfaits et résultats
partiels. Préserver les rosters à l’export et produire un bilan des lignes
rejetées ou exclues.

Critère de réussite : les entrées invalides donnent une erreur localisée ; une
série partielle ne devient pas une cible d’apprentissage finale ; un aller-retour
CSV conserve tous les champs métier ; CSV et API interprètent les mêmes
valeurs de façon identique.

**6. Corriger les handicaps BO5 du catalogue**

Dans [markets.py](../cs2edge/markets.py#L254), `build_markets` étiquette le
sweep « handicap -1.5 » pour les BO3 et les BO5. En BO5, cette ligne couvre
les scores 3-0 et 3-1 ; le sweep seul correspond à -2.5.

Avec cinq maps indépendantes à 50 %, le catalogue annonce **12,5 %** pour
`a -1.5`, alors que la somme `P(3-0) + P(3-1)` vaut **31,25 %**.
Le handicap positif utilise lui aussi le complément du sweep et doit être
aligné sur la ligne affichée.

Changement proposé : soit étiqueter ces lignes avec `need - 0.5`, soit calculer
leur probabilité en additionnant les scores qui couvrent réellement le handicap.
Les picks de `select_analysis_picks` calculent déjà une ligne de sweep adaptée
au format ; le défaut constaté concerne le catalogue `build_markets`.

Critère de réussite : pour chaque format et chaque signe, la probabilité d’une
ligne égale la somme des probabilités des scores qui la couvrent.

**7. Corriger la portée des options CLI**

Dans [cli.py](../cs2edge/cli.py#L354), les options communes sont installées
sur le parseur racine et sur les sous-commandes. Les valeurs par défaut de
la sous-commande écrasent celles déjà lues.

Reproduction avec `build_parser().parse_args(...)` :

```text
--data /tmp/custom.csv --no-gbdt predict navi g2
  -> data='data/sample_matches.csv', no_gbdt=False

predict navi g2 --data /tmp/custom.csv --no-gbdt
  -> data='/tmp/custom.csv', no_gbdt=True
```

Cela touche aussi les options communes `--warmup` et `--elo-config`.
Une commande peut donc analyser le jeu synthétique alors que l’utilisateur
a explicitement demandé son propre fichier.

Changement proposé : définir les valeurs par défaut une seule fois et empêcher
les options absentes de la sous-commande d’écraser celles du parseur racine,
par exemple avec `argparse.SUPPRESS` sur ce second niveau.

Critère de réussite : déplacer chaque option commune avant ou après la
sous-commande produit les mêmes paramètres et utilise le même fichier.

**8. Renforcer les données et la validation du modèle**

Le CSV fourni contient 4 758 maps, 2 416 séries et aucun roster renseigné.
Dans **395 séries, soit 16,35 %**, une même map apparaît plusieurs fois.
Le générateur utilise des tirages avec remise dans
[sources.py](../cs2edge/sources.py#L194), tandis que le veto projeté retire
les maps du pool. Les données de démonstration et la structure évaluée
ne suivent donc pas le même mécanisme de sélection.

Les tests d’apprentissage emploient le régime synthétique par défaut
`facile`, avec quelques graines fixes. Le régime `realiste` disponible dans
le code n’est pas utilisé par ces tests. Le pool `ACTIVE_DUTY` est également
constant sur toute la période. Enfin, les marges de `TipsterRecord.by`
comptent les pronostics, bien que plusieurs pronostics d’un même match
dépendent du même résultat.

Changements proposés :

- Produire les séries simulées avec des maps distinctes, des dates de départ
  fixes, plusieurs graines et les deux régimes ; mesurer l’écart au générateur
  avec vérité latente lorsque c’est pertinent.
- Constituer un jeu réel avec provenance, identifiants stables, dates de
  disponibilité et période finale réservée. Versionner le pool de maps et
  intégrer les effectifs connus avant le match. Le gain de ces informations
  reste une hypothèse à mesurer sur ce jeu.
- Publier les différences de log-loss et de Brier face aux références avec
  des intervalles obtenus par rééchantillonnage des matchs ou blocs temporels.
  Regrouper les marchés corrélés d’une même rencontre lors du calcul des marges.
- Archiver toutes les fixtures présentées, y compris les abstentions, pour
  calculer le taux de couverture réel du journal. Actuellement `scorecard`
  compte les lignes réglées comme des matchs et fixe les abstentions à zéro.
- Ajouter une CI pour les versions de Python prises en charge, le cœur NumPy
  et l’option GBDT, avec les vérifications métier des points précédents. Aucun
  workflow de CI n’est présent dans les fichiers suivis examinés.

Critère de réussite : un rapport reproductible distingue les jeux synthétiques
du jeu réel, mesure un gain sur une période jamais utilisée pour le choisir et
indique le nombre de rencontres distinctes derrière ses marges statistiques.

**Vérifications effectuées et portée**

`python -m tests.test_smoke` : **32/32 tests passent**.

`python -m cs2edge backtest --no-gbdt --by-year` : **1 376 matchs évalués**,
log-loss **0,4285**, Brier **0,1400**, exactitude **79,3 %**. Ces valeurs
proviennent du CSV synthétique fourni et du pipeline actuel décrit plus haut.

Les écarts de garde-fou ont été mesurés en réutilisant les sorties brutes des
mêmes modèles entraînés. Le chevauchement du réglage a été mesuré en traçant
les identifiants renvoyés par les véritables appels à `walk_forward` dans
`tune`, avec une grille réduite. Les cas de journal, d’import et de BO5 ont
été reproduits dans des fichiers temporaires, sans modifier les données fournies.

Scikit-learn est absent de cet environnement : le fonctionnement numérique du
GBDT et sa sauvegarde complète restent à couvrir. L’écart de configuration
GBDT décrit au point 1 a été vérifié sur les objets créés et dans les appels
du code, sans entraîner ce composant.

Après les correctifs de cohérence et de données, un prolongement utile serait
de sauvegarder un analyseur complet : modèles match et maps, configuration Elo,
état des équipes, calibration, version des features et empreinte des données.
Aujourd’hui `train` sauvegarde seulement le modèle match et les commandes de
prédiction réentraînent l’analyseur. Un tel artefact faciliterait la reproduction
exacte d’une annonce et permettrait un cache invalidé explicitement à chaque
changement de données ou de configuration.
