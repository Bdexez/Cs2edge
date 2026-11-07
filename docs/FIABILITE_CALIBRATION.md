# Fiabilité des calculs et calibration

Travaux réalisés après `06cc329`, vérifiés le 11 novembre 2026. Le CSV de
démonstration reste inchangé et synthétique.

| Axe | Défaut reproduit | Changement vérifié |
| --- | --- | --- |
| Retrait de marge | L’exposant était limité à [0,5 ; 2], même quand la racine sortait de cet intervalle | Encadrement adaptatif de la racine ; deux probabilités issues du même exposant, somme un, invariance à l’inversion des cotes |
| Calcul des mises | Une probabilité NaN pouvait produire la mise maximale ; des paramètres invalides arrivaient aux bilans | Contrôles des probabilités, cotes, plafonds, fractions, mises fixes et résultats binaires avant tout calcul |
| Veto et séries | Un pool de cinq maps en BO3 provoquait un IndexError ; une série trop courte créait des scores non terminaux | Erreurs explicites, maps distinctes, formats et probabilités vérifiés ; sigmoid stable pour un momentum extrême et probabilités déterministes préservées |
| Réconciliation | Quarante itérations IPF pouvaient laisser une victoire à 79,61 % pour une cible de 80 % | Résolution du tableau vainqueur × longueur par dichotomie ; les deux marges sont respectées à la précision flottante, y compris aux bornes |
| Calibration de longueur | Les prédictions historiques utilisaient parfois un modèle entraîné sur leurs propres rencontres | Prédictions par blocs, entraînées uniquement sur les jours antérieurs ; mêmes échantillons pour le service et le backtest |
| Momentum | Le calcul ne mesurait pas le coefficient cumulé appliqué par la loi de série et réutilisait le modèle ajusté | Régression logistique avec probabilités antérieures comme offset, bilan cumulé comme variable et variance regroupée par rencontre |
| Persistance | Une calibration correcte aurait exigé de reconstruire ses modèles au chargement | Format v2 avec historique de calibration, dates limites et contrôles d’intégrité ; anciens instantanés v1 toujours lisibles |
| Options de sélection | Un seuil NaN pouvait contourner l’abstention | Mêmes domaines contrôlés en sélection pure, dans Tipster et dans son backtest |

## Protocole de calibration

Les premières rencontres sans 50 maps d’entraînement antérieures restent sans
prédiction de calibration. Ensuite, un modèle de maps logistique prédit un bloc
de 150 rencontres, prolongé jusqu’à la fin du dernier jour. Le bloc suivant
réentraîne sur les maps des jours strictement antérieurs. La calibration interne
de chaque modèle conserve elle aussi ses frontières de journées.

Les frontières partent du début de l’historique : ajouter des résultats futurs
ne modifie pas les anciennes prédictions. Chaque probabilité conserve la date
limite d’entraînement. Les tests vérifient aussi que changer les scores futurs
ne change ni les probabilités passées ni la température calculée avant eux.

La température de longueur utilise au plus les 800 dernières rencontres
disponibles avant la prédiction ou le fold. Sans 60 séries éligibles et deux
issues observées, elle reste à 1. Le momentum nécessite au moins 30 observations
après chaque type de résultat et 20 rencontres distinctes. L’effet n’est appliqué
automatiquement que si son amplitude dépasse deux erreurs types.

La loi de série applique `momentum × (maps gagnées − maps perdues)` avant la
map suivante. L’estimation utilise exactement cette variable. Les scores de
régression sont regroupés par rencontre pour calculer la variance : dupliquer
les observations d’une même rencontre ne réduit pas l’erreur type. Une
séparation des résultats ou un manque d’information n’active pas l’effet.

## Instantanés et compatibilité

```bash
python -m cs2edge train --no-gbdt --bundle models/analyzer.cs2edge
python -m cs2edge slate --model models/analyzer.cs2edge --fixtures data/fixtures.csv --record data/journal.csv
python -m cs2edge tipster-backtest --no-gbdt
```

Les instantanés v2 embarquent les probabilités et leurs dates limites, sous
sommes SHA-256. Les dimensions, domaines, identifiants et antériorité sont
contrôlés au chargement. Une carte future chargée utilise cette calibration
sans réentraîner les classifieurs, même avec les options automatiques par défaut.
L’écriture reste atomique. Une archive v1 reste utilisable pour prédire ; son
historique de calibration est calculé à la première demande correspondante.

Le veto BO3/BO5 implémenté exige un pool de sept maps distinctes. Les probabilités
et paramètres non finis sont désormais refusés. Ces contrôles rendent certaines
entrées auparavant tolérées explicitement invalides. Le format JSON du seul
modèle match reste disponible. Les exigences de confiance et de version
scikit-learn des composants GBDT restent applicables.

## Vérifications et résultats

- Cœur NumPy : **136 tests réussis**, groupe GBDT ignoré s’il n’est pas installé.
- Avec scikit-learn : **141 tests réussis**, dont l’entraînement et la restauration
  d’un véritable GBDT. Environnement local : Python 3.14.8, scikit-learn 1.9.1.
- Exécution de `train --bundle` puis `slate --model` depuis `/tmp`, avec et sans
  GBDT, calibration et momentum automatiques ; archivage des dix fixtures livrées.
- Le momentum mesuré sur le jeu livré vaut environ +0,063, avec une marge de
  deux erreurs types de 0,099 : il reste désactivé. La température vaut 0,977.
- Les deux cartes de démonstration donnent dix pronostics et quatre abstentions,
  soit quatorze lignes pour dix rencontres.

Le backtest du pronostiqueur logistique présente 1 418 rencontres, en écarte 422
et règle 1 413 pronostics. La température réestimée par fold varie de 0,78 à 1,33.
Les sélections vainqueur portent sur 996 rencontres : probabilité moyenne 74,1 %,
fréquence observée 76,0 %. Ces chiffres décrivent le jeu synthétique.

Le backtest des probabilités de victoire conserve une log-loss de 0,5709 et
un Brier de 0,1949 pour le mélange logistique : la présente série modifie surtout
les calculs des marchés et leurs calibrations, sans revendiquer un gain sur
le modèle vainqueur. Les hyperparamètres par défaut sont conservés.

## Limites

Ces tests ne démontrent ni rendement financier ni performance sur un historique
professionnel. La variance du momentum suppose les rencontres indépendantes ;
des dépendances de tournoi peuvent demander des blocs plus larges. Une
association résiduelle ne prouve pas un effet causal et peut refléter des
informations manquantes. La validation sur données réelles, le pool versionné
par date et la prise en compte des compositions disponibles avant les matchs
restent des axes distincts.
