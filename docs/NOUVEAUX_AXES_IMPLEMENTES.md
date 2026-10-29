# Nouveaux axes implémentés

Suite du travail après `5b9c0d2`, sur le même jeu de démonstration. Ces changements
corrigent des défauts reproduits et ajoutent deux fonctions : la réutilisation
d’un analyseur entraîné et une mesure d’incertitude face à la référence Elo.

| Axe | Défaut ou besoin | Résultat et vérification |
| --- | --- | --- |
| Séries de résultats | Une victoire après une défaite redémarrait à zéro | La série redémarre à +1 ou −1 ; test des inversions successives et de la feature produite |
| Symétrie des ratings | Le calcul Glicko réutilisait le complément de l’attente adverse malgré des incertitudes différentes | Chaque équipe utilise l’incertitude de son adversaire ; inverser les équipes conserve les mêmes ratings, avec ou sans marge de victoire |
| Inactivité des maps | Une note de map ancienne gardait presque tout son avantage ; une autre map pouvait rafraîchir sa date | Chaque map possède sa date d’activité et revient vers 1500 selon la demi-vie configurée ; la prochaine mise à jour part de la note amortie |
| Entrées numériques | NumPy pouvait diffuser silencieusement une colonne de probabilités sur une autre dimension | Dimensions, valeurs finies, labels binaires, probabilités, poids et hyperparamètres validés avant calcul ; les probabilités exactes 0 et 1 restent acceptées |
| Configuration Elo | Des constantes nulles ou négatives provoquaient divisions par zéro ou évolutions incohérentes | Contrôles des demi-vies, incertitudes, amplitudes et booléens ; désactiver une mise à jour avec K=0 reste possible |
| Incertitude du backtest | Un petit gain moyen pouvait être surinterprété | Bootstrap apparié par blocs calendaires ; dupliquer les lignes d’un bloc ne rétrécit pas son intervalle ; aucune borne inventée avec un seul bloc |
| Persistance de l’analyseur | Le JSON match seul ne suffisait pas à servir les prédictions complètes | Archive unique avec historique, configuration et deux modèles ; prédictions identiques après restauration, y compris avec un GBDT réel |
| Parcours CLI | Chaque prédiction rechargeait le CSV et réentraînait les modèles | `train --bundle`, puis `predict/slate --model`, fonctionnent depuis un autre répertoire sans CSV source ni réentraînement pour les dates futures |
| Surcharge du momentum | Un zéro explicite pouvait conserver le momentum d’un analyseur chargé | Une valeur explicite, y compris zéro, remplace la valeur sauvegardée et désactive l’estimation automatique historique ; NaN et infini rejetés |
| CI | La sauvegarde complète devait être testée dans les environnements supportés | Tests unitaires et parcours train → predict → slate/record sur Python 3.9 à 3.14 et dans le job GBDT |

## Utilisation

```bash
python -m cs2edge train --no-gbdt --bundle models/analyzer.cs2edge
python -m cs2edge predict navi g2 --model models/analyzer.cs2edge --date 2026-11-11
python -m cs2edge slate --model models/analyzer.cs2edge --fixtures data/fixtures.csv --record data/journal.csv
python -m cs2edge backtest --no-gbdt --bootstrap-samples 2000 --block-days 14
```

L’archive garde les noms et les effectifs exacts de l’historique chargé, les
options d’entraînement, les coefficients et la calibration des deux modèles.
L’état des features est reconstruit depuis cet historique sans réajuster les
modèles pour une date future. Une demande historique conserve la règle des
jours strictement antérieurs et réentraîne sur ce seul passé.

Les schémas de features, le pool et la version du format sont contrôlés avant
restauration. Les composants possèdent des sommes SHA-256 ; une archive altérée,
incomplète, contenant un nom inattendu ou incompatible est refusée. L’écriture
remplace atomiquement le fichier : un échec conserve l’archive précédente.
Ces sommes détectent une modification et n’authentifient pas l’auteur.

Les archives GBDT utilisent pickle et doivent provenir d’une source de
confiance. Elles exigent la même version de scikit-learn ; les archives
logistiques n’exigent pas cette dépendance. Le JSON historique `train --out`
reste disponible pour le seul modèle match. Avec `--model`, les options
`--data`, `--warmup`, `--no-gbdt` et `--elo-config` sont refusées pour éviter
qu’une configuration demandée soit ignorée.

## Vérifications du 10 novembre 2026

Environnement local : Python 3.14.8, NumPy 2.5.4 pour le parcours GBDT et
scikit-learn 1.9.1. Les tests couvrent aussi le parcours sans cette dépendance.
La suite complète passe **118 tests avec GBDT**. Le parcours NumPy passe
113 tests, avec le groupe GBDT explicitement ignoré quand il n’est pas installé.

Le backtest évalue 1 418 rencontres du 20 février 2025 au 7 novembre 2026,
sur dix folds. Même CSV que la série précédente, constantes par défaut conservées :

| Variante | Log-loss | Brier | Gain de log-loss face à Elo, IC 95% |
| --- | --- | --- | --- |
| Mélange logistique | 0,5709 | 0,1949 | +0,0182 ; [+0,0087 ; +0,0272] |
| Mélange avec GBDT match | 0,5708 | 0,1947 | +0,0184 ; [+0,0090 ; +0,0277] |
| Référence Elo | 0,5891 | 0,2013 | — |

Intervalles calculés avec 1 000 tirages déterministes, 90 blocs observés de
sept jours, en comparant les deux modèles sur les mêmes rencontres à chaque
tirage. Le gain positif signifie une log-loss plus faible que celle d’Elo.
Les nouvelles corrections ne constituent pas un gain de score par rapport
aux chiffres précédents ; leur objet est la justesse du calcul.

Les commandes complètes de restauration ont été exécutées depuis `/tmp`,
avec et sans GBDT : prédiction, carte des dix fixtures livrées et archivage.
Avec momentum nul et calibration de longueur désactivée, chaque carte produit
neuf pronostics et quatre abstentions, soit treize lignes pour dix rencontres.

## Limites et pistes restantes

Le jeu reste synthétique : ces scores et intervalles ne prouvent aucune
performance sur les matchs professionnels ni aucun rendement de pari.
Le bootstrap conditionne ses conclusions à cette période et aux modèles
évalués. Sept jours ne garantissent pas de couvrir toutes les dépendances
d’un tournoi ; `--block-days` permet une analyse de sensibilité. Il ne couvre
pas les changements futurs de méta ni le choix des hyperparamètres après examen
du test. Les essais de réglage n’activent pas le bootstrap par défaut.

Un historique professionnel représentatif, un pool versionné par date et
des compositions connues avant les matchs restent les prochains axes de
validation. L’instantané embarque l’historique pour préserver les reconstructions
historiques ; sa taille augmente donc avec le volume de données.
