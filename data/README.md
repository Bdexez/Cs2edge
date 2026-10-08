# Données de démonstration

`sample_matches.csv` est entièrement synthétique : régime `realiste`, 32 équipes,
trois ans, graine 7 et début au 10 novembre 2023. Les noms d’équipes ne représentent
pas leur niveau réel. Le générateur choisit des maps distinctes pour chaque série.
Les rosters ne sont pas renseignés. Le pool de simulation reste fixe ; ce jeu
ne représente pas les rotations historiques du circuit professionnel.

Reproduction exacte du fichier :

```bash
python -m cs2edge sample --start 2023-11-10 --regime realiste --teams 32 --years 3 --seed 7 --out data/sample_matches.csv
python -m cs2edge fixtures --date 2026-11-10 --out data/fixtures.csv
```

`fixtures.csv` contient des affiches fictives sans cotes. Pour une démonstration
à la date du jour, relancer `fixtures` sans `--date`. Pour une utilisation réelle,
fournir son propre historique et ses propres rencontres, avec leurs identifiants
de source dans la colonne optionnelle `match_id`.

Les métriques obtenues sur ces fichiers décrivent le monde simulé. La mesure
sur des matchs professionnels exige un jeu réel avec provenance et dates de
disponibilité, et une période de validation réservée.
