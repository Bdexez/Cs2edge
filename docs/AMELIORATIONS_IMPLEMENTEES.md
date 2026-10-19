# Améliorations implémentées

Travail réalisé à partir de l’[audit initial](AXES_AMELIORATION.md). Les correctifs,
les tests et les données de démonstration sont répartis en commits thématiques.

| Axe | Comportement obtenu | Vérification |
| --- | --- | --- |
| Cohérence des prédictions | Service et deux backtests partagent le contexte de fiabilité et un modèle de maps logistique, même si le GBDT est activé pour les matchs | Comparaison des probabilités du premier fold avec un véritable analyseur entraîné avant le match et avec le pronostiqueur |
| Validation du réglage | Le score final ne compte que les identifiants de la période réservée ; le passé reste disponible pour construire les ratings | Intersection vide entre réglage et validation ; modifier les résultats réservés ne change pas les paramètres retenus |
| Chronologie | Features calculées avant d’intégrer les résultats de la journée ; folds et calibration sur des journées entières | Modifier un résultat du jour ne change pas les features d’un autre match du même jour |
| Prédiction historique | Reconstruction des données, modèles, momentum automatique et calibration à la date demandée | Invariance des probabilités historiques aux résultats futurs |
| Journal | Identifiants de rencontre, rapprochements ambigus laissés en attente, dédoublonnage dans les lots, verrouillage et remplacement atomique | Deux matchs le même jour, équipes inversées, alias, anciens CSV, concurrence et interruption d’écriture |
| Couverture et statistiques | Abstentions archivées ; matchs distincts comptés ; marges prudentes pour les marchés corrélés d’un même match | Dupliquer les marchés d’un match ne réduit pas la marge statistique ; couverture disponible dès l’archivage d’une carte complète |
| Imports et fixtures | CSV et API normalisent les mêmes champs ; scores fractionnaires, formats invalides et cotes non finies rejetés ; effectifs conservés à l’export | Aller-retour des champs métier, erreurs localisées, séries incomplètes exclues des cibles finales et maps dupliquées rejetées |
| Handicaps BO5 | Le −1,5 additionne les scores 3-0 et 3-1 ; le −2,5 correspond au sweep | Chaque probabilité est comparée à la somme des scores qui couvrent la ligne |
| Interface CLI | Les options communes fonctionnent avant et après la sous-commande ; un historique vide produit une erreur explicite | Une véritable commande lit le CSV demandé plutôt que le jeu par défaut |
| Simulations | Maps distinctes, choix du régime et date initiale explicite ; nouveau jeu livré reproductible | Six scénarios couvrent deux régimes et trois graines ; l’estimation de vérité ne modifie pas l’historique généré |
| Artefacts de modèle | Une composante GBDT manquante ou incohérente produit une erreur ; version, dimensions et valeurs des paramètres sont vérifiées | Symétrie de l’ensemble réellement entraîné, prédictions identiques après rechargement, rejet des artefacts altérés |
| CI | Python 3.9 à 3.14, NumPy minimal sur 3.9 et job GBDT dédié | La découverte unittest inclut aussi les 32 scénarios historiques |

**Utilisation et compatibilité**

```bash
python -m unittest discover -s tests -v
python -m cs2edge --data data/sample_matches.csv --no-gbdt backtest --by-year
python -m cs2edge predict navi g2 --no-gbdt --date 2025-06-01
python -m cs2edge slate --no-gbdt --fixtures data/fixtures.csv --record data/journal.csv
python -m cs2edge journal --journal data/journal.csv
```

Le GBDT s’installe avec `pip install -e '.[gbdt]'`. La variable
`CS2EDGE_REQUIRE_GBDT=1` fait échouer sa vérification si la dépendance manque,
au lieu de laisser ce scénario ignoré dans un environnement censé la tester.

Une prédiction sans date utilise aujourd’hui. Une demande historique exclut
tous les résultats du jour demandé, y compris ceux déjà présents dans le fichier.
Cette reconstruction exige suffisamment d’historique pour entraîner le modèle.
Les instantanés de dates récentes sont mis en cache et invalidés au rechargement
ou au réentraînement de l’analyseur.

Les imports deviennent stricts : les fractions ne sont plus tronquées, les maps
nulles ne sont plus supprimées silencieusement et les rencontres incohérentes
produisent une erreur. Une série partielle peut alimenter les ratings avec les
maps déjà connues ; elle n’a pas de vainqueur final et ne règle pas un pronostic.
Les forfaits doivent être traités explicitement par l’adaptateur de la source.

Ajouter `match_id` aux fixtures permet de distinguer deux rencontres entre les
mêmes équipes le même jour. Sans identifiant, le règlement utilise équipes,
format, événement et proximité de date, et laisse une égalité de candidats en
attente. Les anciens journaux restent lisibles ; leur couverture est annoncée
comme indisponible quand les abstentions n’ont pas été archivées.

Le journal ajoute l’heure UTC d’émission, la date limite des données et la version
du logiciel. Il archive les abstentions avec leur motif. Le verrou adjacent
`*.csv.lock` doit rester en place pendant les écritures concurrentes ; les
verrous habituels dans `data/` sont ignorés par Git.

Le catalogue BO5 propose les lignes ±1,5 et ±2,5. Le sens du handicap suit la
marge de maps, et non une assimilation de chaque handicap négatif au sweep.

**Résultats locaux vérifiés le 10 novembre 2026**

Python 3.14.8, NumPy 2.5.4 et scikit-learn 1.9.1 : **91 tests réussis**, dont
les 32 scénarios historiques. Le cœur NumPy a aussi été vérifié séparément,
avec le groupe GBDT ignoré lorsqu’il n’est pas installé.

Sur le nouveau jeu synthétique, 1 418 matchs sont évalués sur dix folds :

| Variante | Log-loss | Brier |
| --- | --- | --- |
| Mélange avec modèles logistiques | 0,5706 | 0,1947 |
| Mélange avec GBDT pour le modèle match | 0,5703 | 0,1945 |
| Référence Elo | 0,5891 | 0,2013 |

Ces résultats décrivent le nouveau jeu de démonstration. Ils ne constituent
pas une comparaison avant/après sur les mêmes données : l’ancien CSV contenait
des séries avec maps répétées et utilisait un autre régime.

Le réglage complet améliore ici la log-loss de validation d’environ 0,0004,
en dessous du seuil de conservation de 0,0005. La commande conserve donc les
paramètres par défaut et n’enregistre pas une configuration sur ce résultat.

Le parcours `slate --record` a également été exécuté avec les fixtures livrées :
dix rencontres présentées, quatre abstentions et quatorze lignes de journal,
dont dix pronostics. La couverture se calcule sur les dix rencontres, pas sur
les quatorze lignes.

La [CI](../.github/workflows/tests.yml) exécute les tests et un backtest dans
chaque environnement. Le job Python 3.9 utilise Ubuntu 22.04, car l’image
Ubuntu 26.04 actuellement associée à `ubuntu-latest` ne fournit pas cette version.

**Prolongements nécessitant une validation supplémentaire**

La validation sur un historique professionnel réel, le pool de maps versionné
par date et les informations de composition disponibles avant les matchs
restent des travaux distincts. Le [jeu livré](../data/README.md) reste synthétique
avec un pool fixe et sans effectifs. Les marges statistiques regroupent les
pronostics d’une rencontre ; les dépendances entre rencontres d’un même tournoi
nécessiteraient une validation par blocs plus larges. La sauvegarde actuelle
porte sur le modèle match, sans constituer un instantané complet de l’analyseur.
