# Fiabilite des entrees et du cycle de vie de l'analyseur

Audit du 11 novembre 2026, a partir de `a039cf7`. Trois axes ont ete reproduits,
corriges et couverts par 18 nouveaux tests de regression.

| Axe | Defaut reproduit | Comportement obtenu |
| --- | --- | --- |
| Dates | `2026-01-01oops` et `2026-01-01T99:99:99` devenaient le 1er janvier | Toute la valeur est validee, avec controle de l'heure et du decalage UTC lorsqu'ils sont fournis |
| CSV | Une equipe absente devenait `none` ; une colonne dupliquee ecrasait sa premiere valeur ; un journal existant vide etait remplace | Lecteur partage par historique, fixtures et journal : rejet des doublons de colonnes, lignes tronquees ou trop longues et guillemets incomplets avant toute reecriture |
| Etat | Un import echoue pouvait melanger nouvel historique et anciens modeles ; un echec du second entrainement conservait deux generations de modeles ; un momentum estime survivait a un changement de donnees | Preparation des donnees, deux modeles et calibration avant remplacement ; preservation de l'analyseur precedent en cas d'erreur ; nouvelle estimation automatique apres changement de donnees ou de parametres |

Les dates historiques conservent leur jour calendaire ecrit : un horodatage muni
d'un fuseau n'est pas converti en UTC. La correction ne change pas la regle
d'exclusion des resultats du jour de la prediction.

Un champ optionnel CSV peut etre vide, mais sa cellule doit exister si sa colonne
figure dans l'entete. Les fichiers UTF-8 avec BOM et les champs cites contenant
virgules ou retours a la ligne restent acceptes. Les en-tetes historiques du
journal restent lisibles. Les tests verifient que les octets d'un journal mal
forme sont preserves apres un ajout ou un pointage refuse.

Un import reussi invalide toujours les modeles : il faut entrainer a nouveau.
Le mode momentum automatique reste actif, mais sa valeur repart de zero jusqu'a
la nouvelle estimation. Un momentum impose explicitement reste conserve. Un
echec de l'estimation automatique pendant le reentrainement preserve lui aussi
les modeles precedents et leur configuration.

Validation locale avec Python 3.14.8 et NumPy 2.5.4 :

```bash
python3 -m unittest discover -s tests -q
# 154 tests, OK, un ignore : scikit-learn absent de cet environnement de base.

# Avec l'option GBDT installee (scikit-learn 1.9.1 lors de cette verification) :
CS2EDGE_REQUIRE_GBDT=1 python3 -m unittest discover -s tests -q
# 159 tests, OK, aucun ignore.

python3 -m cs2edge backtest --no-gbdt --by-year
# 1 418 rencontres evaluees ; log-loss 0,5709, Brier 0,1949.
```

Le parcours CLI `train --bundle` → `predict --model` → `slate --record` →
`settle` → `journal` a aussi ete execute avec et sans GBDT, avec momentum et
calibration automatiques. Dans les deux cas, les 10 fixtures produisent 14 lignes
archivees ; une seconde execution n'ajoute aucun doublon. Les fixtures etant
futures par rapport aux resultats disponibles, ce pointage laisse les pronostics
en attente ; le reglement effectif est couvert par les tests du journal.

Ces metriques proviennent du jeu synthetique fourni. Les changements portent sur
la validation des entrees et la coherence de l'etat ; ils n'etablissent aucun
gain predictif sur des matchs reels.
