<div align="center">

# 🎯 cs2edge

**Pronostiqueur calibré pour le CS2 professionnel.**

Il estime la probabilité de victoire avant le match, *choisit sur quels matchs il se prononce*,
explique pourquoi, archive ses annonces — et publie son propre bilan.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Moteur](https://img.shields.io/badge/moteur-NumPy%20seul-013243?logo=numpy&logoColor=white)
![Tests](https://img.shields.io/badge/tests-32%20OK-2ea44f)
![Probabilites](https://img.shields.io/badge/probabilit%C3%A9s-calibr%C3%A9es-1f6feb)

</div>

---

> [!TIP]
> **Deux règles ont guidé toute la construction.**
> 1. **Des probabilités calibrées, pas des pronostics à l'aveugle.** Quand il annonce 70 %, l'équipe gagne environ 70 fois sur 100 — c'est vérifiable, et c'est vérifié.
> 2. **Se taire vaut mieux que meubler.** Sur les données livrées, il écarte 17 % des matchs. La sélectivité *est* le produit : un pronostiqueur qui se prononce sur tout ne permet pas de distinguer ses convictions de ses devinettes.

### En bref

- 🃏 **Une carte de pronostics** classée par conviction, chaque ligne argumentée par des chiffres.
- 🗳️ **La structure du veto** déroulée map par map — là où les modèles naïfs se trompent le plus.
- 💹 **Deux modes** : « qui gagne » sans cotes, « où le marché se trompe » avec cotes.
- 📊 **Un bilan honnête** : annoncé *contre* observé, marge d'échantillonnage comprise.
- 🧮 **Cœur en NumPy seul** : régression logistique, calibration et pipeline écrits à la main.

---

## Sommaire

- ⚡ [Démarrage rapide](#démarrage-rapide)
- 🃏 [Le pronostiqueur](#le-pronostiqueur)
- 🗳️ [Le veto, et pourquoi il change tout](#le-veto-et-pourquoi-il-change-tout)
- 🔎 [L'analyse d'un match isolé](#lanalyse-dun-match-isolé)
- 📊 [Rendre des comptes](#rendre-des-comptes)
- ⚙️ [Comment ça marche](#comment-ça-marche)
- 📥 [Brancher vos données](#brancher-vos-données)
- 🔬 [Mesurer, et se méfier des chiffres flatteurs](#mesurer-et-se-méfier-des-chiffres-flatteurs)
- 🧪 [Ce que la mesure a corrigé](#ce-que-la-mesure-a-corrigé)
- ⚠️ [Ce que ça ne fait pas](#ce-que-ça-ne-fait-pas)
- 🗂️ [Structure du code](#structure-du-code)

---

## Démarrage rapide

```bash
git clone <ce dépôt> && cd cs2edge
python -m venv .venv && .venv/bin/pip install -e .

# optionnel : ajoute un GBDT à l'ensemble (+ export pandas)
.venv/bin/pip install -e ".[gbdt]"
```

Le cœur ne dépend que de **NumPy** — la régression logistique, la calibration et tout le
pipeline sont écrits à la main. `scikit-learn` est facultatif.

```bash
cs2edge sample                              # historique synthétique pour tester
cs2edge fixtures --out data/fixtures.csv    # matchs à venir (exemple)
cs2edge slate --fixtures data/fixtures.csv  # 🃏 la carte de pronostics
cs2edge tipster-backtest                    # 📊 le bilan du pronostiqueur
cs2edge backtest --by-year                  # 📊 le bilan du modèle
cs2edge ratings --top 15                    # classement Elo
cs2edge predict navi g2 --bo 3 --odds-a 1.75 --odds-b 2.10
python -m tests.test_smoke                  # 32 tests de bout en bout
```

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
  -> handicap -1.5 : falcons   81%   [solide]   cote juste 1.24
       . Pool desequilibre en faveur de falcons : le veto lui laisse ses
         meilleurs terrains, ce qui rend le sweep (2-0) plus probable (81%).
       . Pronostic correle au vainqueur : ne pas cumuler les mises.

[...]

ECARTES (2)
========================================================================
  falcons vs 9z            match trop serre (50%) pour un pronostic assume
  9z vs faze               match trop serre (58%) pour un pronostic assume
```

La carte est **classée par conviction**. Chaque pronostic porte un palier, une cote juste, et
les raisons chiffrées qui l'ont produit — y compris celles qui vont **contre** le pronostic,
parce qu'un raisonnement qui ne mentionne que ce qui l'arrange n'est pas un raisonnement.

### Deux modes, à ne pas confondre

|  | 🎯 sans cotes | 💹 avec cotes |
|---|---|---|
| **question posée** | « qui gagne ? » | « où le marché se trompe-t-il ? » |
| **critère** | probabilité ≥ 60 % | écart ≥ 4 pts après retrait de la marge |
| **on optimise** | le taux de réussite | la valeur attendue |
| **taux de réussite** | élevé | **structurellement plus bas** |

> [!IMPORTANT]
> Cette distinction est le point aveugle de la plupart des pronostiqueurs. En mode marché, la
> valeur se trouve surtout sur des outsiders : on perd plus souvent, et c'est normal. Mettre en
> avant son taux de réussite pour vendre des pronostics à cotes, c'est vendre une illusion.

Remplir **une seule** colonne de cotes d'un fichier de matchs bascule automatiquement en mode
marché — chaque marché est évalué indépendamment.

**En mode marché, deux garde-fous :**

- 🛡️ **Aucun pari à espérance négative.** L'écart se mesure contre la cote *sans sa marge* ;
  l'espérance de gain, elle, contre la cote *réellement offerte*. Sur un marché à forte marge,
  battre la « vraie » probabilité du bookmaker ne suffit pas à battre son prix — le pronostiqueur
  n'annonce donc jamais un pari dont l'EV est négative.
- 🎚️ **Mise cumulée plafonnée par match.** Vainqueur, total de maps et handicap tirent tous du
  même résultat : les additionner à pleine mise Kelly reviendrait à miser deux à trois fois la
  taille voulue sur une position unique. Leur somme est ramenée au plafond d'un pari seul.

---

## Le veto, et pourquoi il change tout

Un bo3 ne se joue pas sur trois maps tirées au hasard : la map 1 est choisie par A, la map 2
par B, et la décisive est ce qui survit aux bans. `cs2edge` déroule cette séquence et en déduit
la **loi complète du score**.

Sur le vainqueur, l'effet est négligeable. Sur les marchés dérivés, il est massif — mesuré sur
un pool polarisé :

| | modèle i.i.d. | avec la structure du veto |
|---|---:|---:|
| vainqueur | 50,0 % | 50,0 % |
| victoire 2-0 | 25,0 % | **12,8 %** |
| série en 3 maps | 50,0 % | **74,5 %** |

C'est exactement là que se trouvent les marchés « total de maps », « handicap » et « score
exact » — donc exactement là où l'approximation coûte cher. Ces cotes sont souvent posées à
partir d'un modèle sans veto.

> [!NOTE]
> Le **handicap -1.5** (le favori gagne sans lâcher une map : un 2-0 en bo3, un 3-0 en bo5) sort
> directement de cette structure. Un pool déséquilibré rend le sweep bien plus probable qu'un
> modèle i.i.d. ne le croit. Le pronostiqueur le porte à son palier de confiance — validé
> jusque dans la bande 0,60–0,70 sur le backtest — et signale qu'il est **corrélé** au
> vainqueur : on ne cumule pas deux mises pleines sur une même conviction.

### L'effet d'enchaînement entre maps (momentum)

Gagner la map précédente change-t-il la suivante, au-delà de ce que le modèle sait déjà des deux
équipes ? `slate` le **mesure** sur l'historique et ne l'applique que s'il dépasse le bruit
(`--momentum auto`, par défaut). C'est la seule quantité laissée à zéro par défaut, précisément
parce qu'elle est facile à halluciner : beaucoup de modèles publics l'imposent à l'intuition, ce
qui revient à inventer du signal.

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

<sub>*(sortie réelle, sur les données synthétiques livrées avec le projet)*</sub>

Avec des cotes, il dit s'il y a un écart exploitable **après retrait de la marge du bookmaker** —
et refuse de conclure quand l'écart est trop beau pour être vrai.

---

## Rendre des comptes

Un pronostiqueur qui n'archive pas ses annonces n'est pas évaluable — et sans trace horodatée,
rien n'empêche de se souvenir surtout des réussites.

```bash
cs2edge slate --fixtures f.csv --record data/journal.csv   # archive AVANT le match
cs2edge settle --journal data/journal.csv                  # pointe APRÈS coup
cs2edge journal --journal data/journal.csv                 # le bilan
```

Le journal est en **ajout seul** : le pointage complète les lignes, il ne les réécrit jamais.
Les deux opérations sont idempotentes.

### Le bilan du pronostiqueur

`cs2edge tipster-backtest` rejoue toute la stratégie sur l'historique, en n'utilisant à chaque
instant que le passé :

```
Bilan du pronostiqueur : 1376 matchs presentes, 235 ecartes (17%), 2147 pronostics emis.

Par palier de confiance (un palier qui ne tient pas sa promesse ment) :
  palier             n   annonce   observe    ecart    marge
  solide          1653    86.0%     87.6%    +1.6%  +/-1.7%
  interessant      494    64.8%     63.8%    -1.1%  +/-4.3%

Par marche :
  marche                   n   annonce   observe    ecart    marge
  handicap -1.5          405    79.3%     82.0%    +2.7%  +/-4.0%
  total maps -2.5        523    77.6%     80.1%    +2.5%  +/-3.6%
  total maps -4.5         56    86.1%     85.7%    -0.4%  +/-9.2%
  vainqueur             1141    83.3%     83.1%    -0.2%  +/-2.2%
```

**Un palier doit tenir sa promesse.** La colonne `ecart` compare l'annoncé au réalisé. Un palier
qui annonce 70 % et réalise 60 % ment à son lecteur, même s'il gagne plus souvent qu'il ne perd.

**Un écart n'est un défaut que s'il dépasse le bruit.** La colonne `marge` donne deux erreurs
types. En dessous, aucune conclusion — c'est le travers classique des bilans de pronostiqueurs,
qui commentent vingt lignes comme si elles démontraient quelque chose.

> C'est ce backtest qui a façonné le produit, pas l'inverse : il a fait supprimer un palier,
> corriger une calibration, et n'ouvrir les marchés dérivés que là où ils tiennent leur promesse.

---

## Comment ça marche

**1. Elo, global et par map.** Une note globale par équipe, plus une note par couple
(équipe, map). Trois raffinements par rapport à un Elo scolaire : marge de victoire amortie
(écraser un outsider rapporte peu), K modulé par le tier de l'événement, et régression vers la
moyenne en cas d'inactivité ou de changement de roster.

**2. Features, en une seule passe chronologique.** Chaque ligne est calculée avec l'état des
équipes *avant* le match, puis l'état est mis à jour. Le même objet sert à l'entraînement et à
la prédiction en production — ce qui élimine les divergences train/serve. Au menu : écart Elo,
avantage de pool, forme pondérée par récence, écart de rounds, historique direct, jours de
repos, charge récente, force du calendrier, série en cours, ancienneté du cinq.

**3. Deux estimateurs indépendants.**
- Un **modèle match** : régression logistique ridge sans constante, résolue par IRLS, plus un
  GBDT optionnel.
- Une **lecture par maps** : probabilité de gagner chacune des 7 maps de l'active duty, dont on
  retire la meilleure et la pire (chacun bannit ce qui l'arrange), puis conversion en série.

Ils sont mélangés à parts égales. Ce poids n'est pas choisi à l'intuition : il est mesuré par
backtest, et `cs2edge backtest` affiche l'optimum sur *vos* données.

<details>
<summary><b>4. Deux garde-fous structurels</b> — antisymétrie & décorrélation</summary>

<br>

*Antisymétrie.* Toutes les features sont des différences (A − B), le modèle n'a pas de constante,
et il est entraîné sur les données miroir `(X, y)` et `(−X, 1−y)`. Résultat :
`p(A bat B) + p(B bat A) = 1` à la précision machine. Sans cela, l'ordre dans lequel on saisit
les équipes déplace la prédiction — c'est le bug silencieux classique de ce genre de système.

*Décorrélation.* Forme, pool, calendrier sont des quasi-doublons de l'écart Elo. Sans retrait, la
pénalité ridge répartit ce signal commun au hasard entre les colonnes et `elo_diff` peut ressortir
avec un poids *négatif*. Le modèle projette donc chaque colonne hors de l'écart Elo —
transformation linéaire, qui ne change pas ce que le modèle peut représenter, seulement la
lisibilité des coefficients.

</details>

**5. Du niveau map au niveau série.** Le veto est déroulé (chacun bannit son plus mauvais
terrain, choisit son meilleur), ce qui donne les maps effectivement jouées et leur ordre. On en
déduit la loi complète du score par énumération.

**6. Deux marginales calibrées, réconciliées.** Le vainqueur vient du modèle match ; la longueur
de la série est calibrée séparément sur l'historique. Un ajustement proportionnel itératif impose
les deux marges à la loi du score sans casser sa structure interne. Sans cette étape, le
pronostic « vainqueur » et le pronostic « score exact » pourraient se contredire — défaut
immédiatement visible pour un lecteur attentif.

**7. Calibration par température.** Un seul paramètre, ajusté sur un holdout temporel. Il préserve
l'ordre et la symétrie, et corrige la sur-confiance systématique. Une probabilité mal calibrée est
pire qu'inutile dès qu'on la compare à une cote.

---

## Brancher vos données

Le projet ne scrape rien : il lit un CSV **au niveau map**, ce qui le rend agnostique de la
source (export HLTV, bo3.gg, Liquipedia, dataset Kaggle, base maison).

```csv
match_id,date,team_a,team_b,map,rounds_a,rounds_b,best_of,event,tier,lan,map_index
2380145,2026-08-14,Vitality,Spirit,inferno,13,9,3,IEM Cologne,s,1,0
2380145,2026-08-14,Vitality,Spirit,nuke,10,13,3,IEM Cologne,s,1,1
```

| colonne | requis | rôle |
|---|:---:|---|
| `match_id` | ✅ | regroupe les maps d'une même série |
| `date`, `team_a`, `team_b`, `map`, `rounds_a`, `rounds_b` | ✅ | — |
| `best_of` | — | défaut 3 |
| `tier` | — | `s/a/b/c/d` — pondère le poids du résultat dans l'Elo |
| `lan` | — | LAN vs online |
| `map_index` | — | ordre des maps dans la série |
| `roster_a`, `roster_b` | — | pseudos séparés par des virgules — détecte les changements d'effectif |

### Les matchs à venir

```csv
date,team_a,team_b,best_of,lan,event,odds_a,odds_b,odds_over,odds_under,odds_a_sweep,odds_b_sweep
2026-09-14,Vitality,Spirit,3,1,IEM Cologne,1.75,2.10,1.95,1.85,3.40,6.50
```

Seules `date`, `team_a`, `team_b` sont requises. Les cotes sont facultatives et font basculer du
mode analyse au mode marché. `odds_a_sweep` / `odds_b_sweep` sont les cotes du **handicap de
maps** (l'équipe gagne sans lâcher une map) : c'est le marché où l'edge du modèle est le plus
grand, parce que les books l'ouvrent souvent depuis un modèle i.i.d. aveugle au veto. Faute de la
cote `+1.5` complémentaire pour retirer la marge, l'edge y est mesuré contre la cote brute — une
lecture volontairement prudente.

Les noms d'équipes sont normalisés automatiquement (`Natus Vincere` → `navi`, `de_Dust2` →
`dust2`). Depuis du code, `sources.from_records()` accepte directement des dicts issus d'une API.

> [!TIP]
> `roster_a`/`roster_b` sont absents de la plupart des exports publics, et c'est dommage : un
> changement de deux joueurs invalide une bonne partie de l'historique d'une équipe, et le modèle
> sait en tenir compte quand on lui donne l'information.

---

## Mesurer, et se méfier des chiffres flatteurs

```bash
cs2edge backtest --by-year
```

Le backtest est **walk-forward** : entraînement sur le passé uniquement, prédiction du bloc
suivant, réentraînement, on avance. C'est le seul protocole honnête ici. Une validation croisée
aléatoire donnerait des scores nettement meilleurs et entièrement faux, parce que la note Elo
d'un match donné encode déjà des matchs postérieurs.

Le rapport affiche le **log-loss** et un `skill score` (réduction du log-loss face au pile ou
face), comparés à trois références : modèle match seul, lecture par maps seule, et Elo brut.

> [!WARNING]
> **L'accuracy est un mauvais indicateur.** L'Elo brut affiche souvent une accuracy *supérieure*
> au modèle complet tout en ayant un log-loss nettement pire : il désigne un peu mieux le
> vainqueur, mais avec des probabilités mal calibrées — donc inutilisables face à un marché. Ce
> sont le **log-loss** et l'**ECE** qui comptent.

Le rapport affiche aussi une courbe de fiabilité : pour chaque tranche de probabilité annoncée, la
fréquence réellement observée. Un écart systématique de plus de 5 points dans une tranche bien
peuplée signale un problème de calibration.

---

## Ce que la mesure a corrigé

Ces décisions ne viennent pas d'une intuition sur le CS, mais d'un chiffre qui disait que quelque
chose n'allait pas.

**🗑️ Un palier a été supprimé.** Un troisième niveau « serré » couvrait la bande 57–60 %. Sur
l'historique : 62 pronostics, 58,6 % annoncés, 66,1 % observés. Une bande de trois points, un
échantillon minuscule, un écart entièrement compatible avec le bruit — du hasard présenté comme un
jugement. Deux paliers qui tiennent valent mieux que trois dont un ment.

**🔧 Un paramètre a été essayé puis retiré.** Le marché « total de maps » annonçait 76,9 % et
réalisait 87,4 %. Première tentative : un paramètre de « dispersion ». Instable d'un fold à l'autre
(0,10 à 1,05) sans corriger le défaut mesuré. Le vrai coupable était ailleurs : la loi du score
était ancrée sur une probabilité de vainqueur volontairement prudente, ce qui tirait toute la
distribution vers l'équilibre. La bonne réponse n'était pas de déformer le veto, mais de calibrer
la seconde marginale comme on calibrait déjà la première. Après correction : **+2,0 % d'écart au
lieu de +10,5 %**.

**➕ Les marchés dérivés ont été ouverts plus largement.** Total de maps et handicap étaient
réservés au palier solide (≥ 0,70). Mesuré sur douze mondes simulés, la bande 0,60–0,70 tient sa
promesse (total −0,5 %, handicap +1,3 %) : les émettre dès le seuil d'abstention a **assaini le
palier « intéressant »** (de −5,1 % à −1,1 %) et augmenté la couverture de **~40 %**, sans qu'aucun
marché ne sorte de sa marge.

---

## Ce que ça ne fait pas

> [!WARNING]
> **Les données livrées sont synthétiques.** `cs2edge sample` simule un historique crédible, ce qui
> permet de faire tourner toute la chaîne sans données réelles — mais un monde simulé est
> **beaucoup plus prévisible** que le vrai. Les ~78 % d'accuracy obtenus dessus ne veulent rien
> dire. Sur des données réelles, attendez-vous à **62–68 % d'accuracy** et un skill score de
> **0,10–0,20**. Un chiffre nettement au-dessus est un signe de fuite temporelle, pas de réussite.

**Le plafond est bas, et il est structurel.** Le CS2 pro est un sport à forte variance : une map
se joue en 24 rounds, un bo3 se retourne sur deux duels. Une partie irréductible du résultat n'est
pas prévisible à partir de l'historique, quel que soit le modèle.

**Le modèle ignore ce qui n'est pas dans les résultats passés** — remplaçant de dernière minute,
joueur malade, problème de visa, transfert annoncé la veille, enjeu sportif nul, décalage horaire,
panne matérielle. C'est pour cela que `predict` plafonne la confiance à 96 %, rétrécit les
probabilités quand l'historique est mince ou périmé, et vous avertit en cas de désaccord massif
avec le marché — dans ce cas, c'est presque toujours le marché qui sait quelque chose.

**Un taux de réussite élevé ne prouve rien.** Annoncer que le premier mondial bat le trentième, et
avoir raison 90 % du temps, ne demande aucun modèle. Le bilan affiche donc *annoncé contre observé*
plutôt qu'un taux brut : la seule question intéressante est de savoir si le chiffre annoncé était
juste.

**Le marché est un adversaire sérieux.** Les cotes de clôture des gros bookmakers sont, sur le long
terme, mieux calibrées que ce que produit ce genre de modèle. Les fonctions de `odds.py` (retrait
de marge par puissance, valeur attendue, Kelly fractionnaire plafonné au quart) rendent la
comparaison honnête — elles ne promettent pas un rendement.

---

## Structure du code

```
cs2edge/
  schema.py     modèle de données, normalisation des noms, lecture CSV
  sources.py    entrées de données + simulateur d'historique
  ratings.py    Elo global et par map (marge, tier, inactivité, roster)
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
tests/          32 tests de bout en bout, sans dépendance externe
```

Les règles de décision du pronostiqueur vivent dans une **fonction pure**, appelée à la fois par
le service en direct et par son propre backtest. Une logique de décision écrite en deux
exemplaires finit toujours par diverger, et le bilan publié devient faux sans que personne ne
s'en aperçoive.

---

<div align="center">

### Sur l'usage

</div>

Cet outil est un **instrument de mesure** : il quantifie ce que l'historique permet de dire, et il
est explicite sur ce qu'il ignore. Ses pronostics valent pour l'analyse d'un match — ils ne
constituent ni un conseil ni une garantie, et un modèle calibré ne supprime pas la variance : il
la chiffre.

> Un pronostic à 70 % se trompe trois fois sur dix. C'est le comportement attendu, pas une
> anomalie — et c'est aussi pourquoi un pronostiqueur ne se juge que sur des centaines d'annonces,
> jamais sur la dernière.
