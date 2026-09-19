# cs2edge

Pronostiqueur pour le CS2 professionnel. Il estime la probabilité de victoire de
chaque équipe avant le match, **choisit sur quels matchs il se prononce**,
explique pourquoi, archive ses annonces, et publie son propre bilan.

Deux contraintes ont guidé la construction :

1. **Des probabilités calibrées, pas des pronostics à l'aveugle.** Quand il
   annonce 70 %, l'équipe doit gagner environ 70 fois sur 100 — c'est vérifiable,
   et c'est vérifié.
2. **Se taire vaut mieux que meubler.** Sur les données livrées il écarte 17 %
   des matchs. La sélectivité *est* le produit : un pronostiqueur qui se prononce
   sur tout ne permet pas de distinguer ses convictions de ses devinettes.

---

## Le pronostiqueur

```
$ cs2edge slate --fixtures data/fixtures.csv --record data/journal.csv

PRONOSTICS (8 matchs)
========================================================================

falcons vs heroic   bo3, LAN   [Exemple]
  2026-09-09
  -> vainqueur : falcons   92%   [solide]   cote juste 1.09
       . Ecart de niveau : 219 points Elo en faveur de falcons (1807 contre 1588).
       . Le niveau des adversaires affrontes penche vers heroic.
       . Terrain projete apres veto (lecture brute) : anubis (pick falcons, 92%),
         nuke (pick heroic, 90%), dust2 (decisive, 90%).
       . Pool desequilibre : 84% de chances que la serie se termine sans
         derniere map.
  -> total maps -2.5 : oui   84%   [solide]   cote juste 1.19
       . Loi du score issue du veto projete : 2-0 81%, 2-1 11%, 1-2 5%, 0-2 3%.

falcons vs sangal   bo3, LAN   [Exemple]
  2026-09-07
  -> vainqueur : falcons   77%   [solide]   cote juste 1.30
       . Ecart de niveau : 101 points Elo en faveur de falcons (1808 contre 1707).
       . Le niveau des adversaires affrontes penche vers sangal.
       . Terrain projete apres veto (lecture brute) : inferno (pick falcons, 81%),
         dust2 (pick sangal, 73%), mirage (decisive, 78%).
       . Pool desequilibre : 64% de chances que la serie se termine sans
         derniere map.

[...]

ECARTES (2)
========================================================================
  falcons vs 9z            match trop serre (50%) pour un pronostic assume
  9z vs faze               match trop serre (58%) pour un pronostic assume
```

La carte est classée par conviction. Chaque pronostic porte un palier, une cote
juste, et les raisons chiffrées qui l'ont produit — y compris celles qui vont
**contre** le pronostic, parce qu'un raisonnement qui ne mentionne que ce qui
l'arrange n'est pas un raisonnement.

### Deux modes, à ne pas confondre

|  | sans cotes | avec cotes |
|---|---|---|
| question posée | « qui gagne ? » | « où le marché se trompe-t-il ? » |
| critère | probabilité ≥ 60 % | écart ≥ 4 pts après retrait de la marge |
| ce qu'on optimise | taux de réussite | valeur attendue |
| taux de réussite | élevé | **structurellement plus bas** |

Cette distinction est le point aveugle de la plupart des pronostiqueurs. En mode
marché, la valeur se trouve surtout sur des outsiders : on perd plus souvent, et
c'est normal. Quiconque met en avant son taux de réussite pour vendre des
pronostics à cotes vend une illusion.

Remplir les colonnes `odds_a`/`odds_b` d'un fichier de matchs bascule
automatiquement d'un mode à l'autre.

---

## Le veto, et pourquoi il change tout

Un bo3 ne se joue pas sur trois maps tirées au hasard : la map 1 est choisie
par A, la map 2 par B, et la décisive est ce qui survit aux bans. `cs2edge`
déroule cette séquence et en déduit la loi complète du score.

Sur le vainqueur, l'effet est négligeable. Sur les marchés dérivés, il est
massif — mesuré sur un pool polarisé :

| | modèle i.i.d. | avec la structure du veto |
|---|---|---|
| vainqueur | 50,0 % | 50,0 % |
| victoire 2-0 | 25,0 % | **12,8 %** |
| série en 3 maps | 50,0 % | **74,5 %** |

C'est exactement là que se trouvent les marchés « total de maps » et « score
exact », donc exactement là où l'approximation coûte cher. Les cotes de total de
maps sont souvent posées à partir d'un modèle sans veto.

---

## L'analyse d'un match isolé

```
$ cs2edge predict 9z faze --bo 3 --lan

9z  vs  faze   [bo3, LAN, 2026-09-06]

  9z                58.6%   cote juste 1.71
  faze              41.4%   cote juste 2.41

  Favori : 9z (58.6%) - confiance correcte
  Modele match 52.4% | reconstitue par maps 64.8% (vu de 9z)

  Ce qui pese (contribution au logit, vu de 9z) :
    round_form_diff        -1.006  ############
    perf_residual_diff     +0.622  #######
    pool_mean_diff         +0.179  ##
    elo_diff               +0.160  #
    form_diff              +0.125  #
    trajectory_diff        -0.109  #

  Pool de maps (avantage en points Elo, vu de 9z) :
    dust2       +280.7   p=71.7%   [55v30 maps]
    train       +234.1   p=66.6%   [56v60 maps]
    ancient      +84.3   p=59.6%   [54v51 maps]
    anubis       +48.5   p=61.1%   [53v45 maps]
    mirage       +22.8   p=54.2%   [30v57 maps]
    nuke         +21.5   p=53.7%   [47v59 maps]
    inferno      -57.9   p=47.7%   [51v49 maps]

  Historique : 9z 346 maps (derniere 2026-09-01), faze 351 maps
               (derniere 2026-09-06), confrontations directes 12
```

*(sortie réelle, sur les données synthétiques livrées avec le projet)*

Avec des cotes, il dit s'il y a un écart exploitable **après retrait de la marge
du bookmaker** — et refuse de conclure quand l'écart est trop beau pour être vrai.

---

## Rendre des comptes

Un pronostiqueur qui n'archive pas ses annonces n'est pas évaluable — et sans
trace horodatée, rien n'empêche de se souvenir surtout des réussites.

```bash
cs2edge slate --fixtures f.csv --record data/journal.csv   # archive avant le match
cs2edge settle --journal data/journal.csv                  # pointe après coup
cs2edge journal --journal data/journal.csv                 # le bilan
```

Le journal est en ajout seul : le pointage complète les lignes, il ne les
réécrit jamais. Les deux opérations sont idempotentes.

### Le bilan du pronostiqueur

`cs2edge tipster-backtest` rejoue toute la stratégie sur l'historique, en
n'utilisant à chaque instant que le passé :

```
Bilan du pronostiqueur : 1376 matchs presentes, 235 ecartes (17%), 1549 pronostics emis.

Par palier de confiance (un palier qui ne tient pas sa promesse ment) :
  palier             n   annonce   observe    ecart    marge
  solide          1327    86.7%     87.8%    +1.1%  +/-1.9%
  interessant      222    65.0%     59.9%    -5.1%  +/-6.4%

Par marche :
  marche                   n   annonce   observe    ecart    marge
  total maps -2.5        359    83.8%     85.8%    +2.0%  +/-3.9%
  total maps -4.5         49    89.1%     85.7%    -3.3%  +/-8.9%
  vainqueur             1141    83.3%     83.1%    -0.2%  +/-2.2%
```

Deux principes dans ce tableau :

**Un palier doit tenir sa promesse.** La colonne `ecart` compare ce qui a été
annoncé à ce qui s'est produit. Un palier qui annonce 70 % et réalise 60 % ment
à son lecteur, même s'il gagne plus souvent qu'il ne perd.

**Un écart n'est un défaut que s'il dépasse le bruit.** La colonne `marge` donne
deux erreurs types. En dessous, aucune conclusion — c'est le travers classique
des bilans de pronostiqueurs, qui commentent vingt lignes comme si elles
démontraient quelque chose.

C'est ce backtest qui a façonné le produit, pas l'inverse. Il a fait supprimer un
palier (voir plus bas) et corriger une calibration.

---

## Installation

```bash
git clone <ce dépôt> && cd cs2edge
python -m venv .venv && .venv/bin/pip install -e .

# optionnel : ajoute un GBDT à l'ensemble
.venv/bin/pip install -e ".[gbdt]"
```

Le cœur ne dépend que de **numpy** : la régression logistique, la calibration et
tout le pipeline sont écrits à la main. `scikit-learn` est facultatif.

```bash
cs2edge sample                              # historique synthétique pour tester
cs2edge fixtures --out data/fixtures.csv    # matchs à venir (exemple)
cs2edge slate --fixtures data/fixtures.csv  # la carte de pronostics
cs2edge tipster-backtest                    # le bilan du pronostiqueur
cs2edge backtest --by-year                  # le bilan du modèle
cs2edge ratings --top 15
cs2edge predict navi g2 --bo 3 --odds-a 1.75 --odds-b 2.10
python -m tests.test_smoke
```

---

## Brancher vos données

Le projet ne scrape rien : il lit un CSV **au niveau map**, ce qui le rend
agnostique de la source (export HLTV, bo3.gg, Liquipedia, dataset Kaggle, base
maison).

```csv
match_id,date,team_a,team_b,map,rounds_a,rounds_b,best_of,event,tier,lan,map_index
2380145,2026-08-14,Vitality,Spirit,inferno,13,9,3,IEM Cologne,s,1,0
2380145,2026-08-14,Vitality,Spirit,nuke,10,13,3,IEM Cologne,s,1,1
```

| colonne | requis | rôle |
|---|---|---|
| `match_id` | oui | regroupe les maps d'une même série |
| `date`, `team_a`, `team_b`, `map`, `rounds_a`, `rounds_b` | oui | — |
| `best_of` | non | défaut 3 |
| `tier` | non | `s/a/b/c/d` — pondère le poids du résultat dans l'Elo |
| `lan` | non | LAN vs online |
| `map_index` | non | ordre des maps dans la série |
| `roster_a`, `roster_b` | non | pseudos séparés par des virgules — permet de détecter les changements d'effectif |

### Les matchs à venir

```csv
date,team_a,team_b,best_of,lan,event,odds_a,odds_b,odds_over,odds_under
2026-09-14,Vitality,Spirit,3,1,IEM Cologne,1.75,2.10,1.95,1.85
```

Seules `date`, `team_a`, `team_b` sont requises. Les colonnes de cotes sont
facultatives et font basculer du mode analyse au mode marché.

Les noms d'équipes sont normalisés automatiquement (`Natus Vincere` → `navi`,
`de_Dust2` → `dust2`). Depuis du code, `sources.from_records()` accepte
directement des dicts issus d'une API.

`roster_a`/`roster_b` sont absents de la plupart des exports publics et c'est
dommage : un changement de deux joueurs invalide une bonne partie de
l'historique d'une équipe, et le modèle sait en tenir compte quand on lui donne
l'information.

---

## Comment ça marche

**1. Elo, global et par map.** Une note globale par équipe, plus une note par
couple (équipe, map). Trois raffinements par rapport à un Elo scolaire : marge
de victoire amortie (écraser un outsider rapporte peu), K modulé par le tier de
l'événement, et régression vers la moyenne en cas d'inactivité ou de changement
de roster.

**2. Features, en une seule passe chronologique.** Chaque ligne est calculée
avec l'état des équipes *avant* le match, puis l'état est mis à jour. Le même
objet sert à l'entraînement et à la prédiction en production, ce qui élimine
les divergences train/serve. Au menu : écart Elo, avantage de pool de maps,
forme pondérée par récence, écart de rounds, historique direct, jours de repos,
charge de travail récente, force du calendrier, série en cours, ancienneté du
cinq.

**3. Deux estimateurs indépendants.**
- Un **modèle match** : régression logistique ridge sans constante, résolue par
  IRLS, plus un GBDT optionnel.
- Une **lecture par maps** : probabilité de gagner chacune des 7 maps de
  l'active duty, dont on retire la meilleure et la pire (chacun bannit ce qui
  l'arrange), puis conversion en probabilité de série.

Ils sont mélangés à parts égales. Ce poids n'est pas choisi à l'intuition : il
est mesuré par backtest, et `cs2edge backtest` affiche l'optimum sur *vos*
données.

**4. Deux garde-fous structurels.**

*Antisymétrie.* Toutes les features sont des différences (A − B), le modèle n'a
pas de constante, et il est entraîné sur les données miroir `(X, y)` et
`(−X, 1−y)`. Résultat : `p(A bat B) + p(B bat A) = 1` à la précision machine.
Sans cela, l'ordre dans lequel on saisit les équipes déplace la prédiction —
c'est le bug silencieux classique de ce genre de système.

*Décorrélation.* Forme, pool, calendrier sont des quasi-doublons de l'écart Elo.
Sans retrait, la pénalité ridge répartit ce signal commun au hasard entre les
colonnes et `elo_diff` peut ressortir avec un poids *négatif*. Le modèle projette
donc chaque colonne hors de l'écart Elo — transformation linéaire, qui ne change
pas ce que le modèle peut représenter, seulement la lisibilité des coefficients.

**5. Du niveau map au niveau série.** Le veto est déroulé (chacun bannit son
plus mauvais terrain, choisit son meilleur), ce qui donne les maps effectivement
jouées et leur ordre. On en déduit la loi complète du score par énumération.

**6. Deux marginales calibrées, réconciliées.** Le vainqueur vient du modèle
match ; la longueur de la série est calibrée séparément sur l'historique. Un
ajustement proportionnel itératif impose les deux marges à la loi du score sans
casser sa structure interne. Sans cette étape, le pronostic « vainqueur » et le
pronostic « score exact » issus du même match pourraient se contredire — défaut
immédiatement visible pour un lecteur attentif.

**7. Calibration par température.** Un seul paramètre, ajusté sur un holdout
temporel. Il préserve l'ordre et la symétrie, et corrige la sur-confiance
systématique. Une probabilité mal calibrée est pire qu'inutile dès qu'on la
compare à une cote.

---

## Mesurer, et se méfier des chiffres flatteurs

```bash
cs2edge backtest --by-year
```

Le backtest est **walk-forward** : entraînement sur le passé uniquement,
prédiction du bloc suivant, réentraînement, on avance. C'est le seul protocole
honnête ici. Une validation croisée aléatoire donnerait des scores nettement
meilleurs et entièrement faux, parce que la note Elo d'un match donné encode
déjà des matchs postérieurs.

Le rapport affiche le **log-loss** et un `skill score` (réduction du log-loss
face au pile ou face), le tout comparé à trois références : modèle match seul,
lecture par maps seule, et Elo brut.

**L'accuracy est un mauvais indicateur** et le backtest le montre bien : l'Elo
brut affiche souvent une accuracy *supérieure* au modèle complet tout en ayant
un log-loss nettement pire. Il désigne un peu mieux le vainqueur, mais avec des
probabilités mal calibrées — donc inutilisables face à un marché. C'est le
log-loss et l'ECE qui comptent.

Le rapport affiche aussi une courbe de fiabilité : pour chaque tranche de
probabilité annoncée, la fréquence réellement observée. Un écart systématique de
plus de 5 points dans une tranche bien peuplée signale un problème de calibration.

---

## Trois choses que la mesure a corrigées

Ces décisions ne viennent pas d'une intuition sur le CS, mais d'un chiffre qui
disait que quelque chose n'allait pas.

**Un palier a été supprimé.** Un troisième niveau « serré » couvrait la bande
57–60 %. Sur l'historique : 62 pronostics, 58,6 % annoncés, 66,1 % observés. Une
bande de trois points, un échantillon minuscule, un écart entièrement compatible
avec le bruit — autrement dit du hasard présenté comme un jugement. Deux paliers
qui tiennent valent mieux que trois dont un ment.

**Un paramètre a été essayé puis retiré.** Le marché « total de maps » annonçait
76,9 % et réalisait 87,4 %. Première tentative : un paramètre de « dispersion »
corrigeant l'hypothèse de veto parfait. Il s'est révélé instable d'un fold à
l'autre (0,10 à 1,05) sans corriger le défaut mesuré — signe qu'il ne pointait
pas la bonne cause. Un paramètre qui n'améliore pas ce qu'il vise ne mérite pas
de rester dans le code.

Le vrai coupable était ailleurs : la loi du score était ancrée sur une
probabilité de vainqueur volontairement prudente, ce qui tirait toute la
distribution vers l'équilibre. La réponse correcte n'était pas de déformer le
veto mais de calibrer la seconde marginale comme on calibrait déjà la première.
Après correction : **+2,0 % d'écart au lieu de +10,5 %**, et tous les marchés
dans leur marge d'échantillonnage.

**Un avertissement a été recalibré.** La mise en garde « le modèle match et la
lecture par maps divergent » se déclenchait à 15 points d'écart, soit sur un
match sur cinq. Un avertissement qui s'affiche partout n'avertit de rien : seuil
relevé à 25 points, le 95ᵉ centile observé.

---

## Ce que ça ne fait pas, et ce qu'il faut en attendre

**Les données livrées sont synthétiques.** `cs2edge sample` simule un historique
crédible (niveaux latents qui dérivent, forces par map, remaniements d'effectif,
scores produits round par round). Cela permet de faire tourner et tester toute
la chaîne sans données réelles — mais un monde simulé est **beaucoup plus
prévisible** que le vrai. Les ~78 % d'accuracy obtenus dessus ne veulent rien
dire. Sur des données réelles, attendez-vous à **62–68 % d'accuracy** et un
skill score de **0,10–0,20**. Un chiffre nettement au-dessus sur des données
réelles est un signe de fuite temporelle, pas de réussite.

**Le plafond est bas, et il est structurel.** Le CS2 professionnel est un sport
à forte variance : une map se joue en 24 rounds, un bo3 se retourne sur deux
duels. Une partie irréductible du résultat n'est pas prévisible à partir de
l'historique, quel que soit le modèle.

**Le modèle ignore ce qui n'est pas dans les résultats passés :** remplaçant de
dernière minute, joueur malade, problème de visa, transfert annoncé la veille,
enjeu sportif nul (équipe déjà qualifiée), décalage horaire, panne matérielle.
C'est pour cela que `predict` plafonne la confiance à 96 % et rétrécit les
probabilités quand l'historique est mince ou périmé, et qu'il vous avertit
quand il est en désaccord massif avec le marché — dans ce cas, c'est presque
toujours le marché qui sait quelque chose que le modèle ignore.

**Un taux de réussite élevé ne prouve rien.** Annoncer que le premier mondial
bat le trentième, et avoir raison 90 % du temps, ne demande aucun modèle. C'est
pour cela que le bilan affiche systématiquement *annoncé contre observé* plutôt
qu'un taux de réussite brut : la seule question intéressante est de savoir si le
chiffre annoncé était juste.

**Le marché est un adversaire sérieux.** Les cotes de clôture des gros
bookmakers sont, sur le long terme, mieux calibrées que ce que produit ce genre
de modèle. Battre les cotes d'ouverture sur des marchés peu liquides est une
chose ; battre les cotes de clôture en est une autre. Les fonctions de
`odds.py` (retrait de marge par puissance, valeur attendue, Kelly fractionnaire
plafonné à un quart) sont là pour rendre cette comparaison honnête, pas pour
promettre un rendement.

**Un backtest de paris n'a de sens qu'avec de vraies cotes historiques.**
`backtest.bet_backtest()` accepte un dictionnaire `match_id → (cote_a, cote_b)`.
Sans cotes réellement disponibles à l'époque du match, ne reconstituez pas des
cotes moyennes : le biais serait total et le ROI obtenu, une fiction.

---

## Structure

```
cs2edge/
  schema.py     modèle de données, normalisation des noms, lecture CSV
  sources.py    entrées de données + simulateur d'historique
  ratings.py    Elo global et par map (marge de victoire, tier, inactivité, roster)
  features.py   extraction sans fuite temporelle, construction du dataset
  model.py      régression logistique IRLS, décorrélation, calibration, ensemble
  metrics.py    log-loss, Brier, ECE, courbe de fiabilité
  backtest.py   walk-forward, comparaison des estimateurs
  odds.py       retrait de marge, valeur attendue, Kelly
  markets.py    veto, loi du score, marchés dérivés, calibration des marges
  predict.py    orchestration et rapport de prédiction
  pronostic.py  sélectivité, paliers, argumentaire, backtest du pronostiqueur
  journal.py    archivage des annonces, pointage, bilan
  cli.py        interface en ligne de commande
tests/          27 tests de bout en bout, sans dépendance externe
```

Les règles de décision du pronostiqueur vivent dans une fonction pure, appelée
à la fois par le service en direct et par son propre backtest. Une logique de
décision écrite en deux exemplaires finit toujours par diverger, et le bilan
publié devient faux sans que personne ne s'en aperçoive.

---

## Sur l'usage

Cet outil est un instrument de mesure : il quantifie ce que l'historique permet
de dire, et il est explicite sur ce qu'il ignore. Ses pronostics valent pour
l'analyse d'un match — ils ne constituent ni un conseil ni une garantie, et un
modèle calibré ne supprime pas la variance : il la chiffre.

Un pronostic à 70 % se trompe trois fois sur dix. C'est le comportement attendu,
pas une anomalie — et c'est aussi la raison pour laquelle un pronostiqueur ne
doit être jugé que sur des centaines d'annonces, jamais sur la dernière.
