"""Journal des pronostics : emettre, archiver, pointer, rendre des comptes.

Un pronostiqueur qui n'archive pas ce qu'il a annonce n'est pas evaluable, et
ce n'est pas un detail de confort : sans trace horodatee, rien n'empeche de se
souvenir surtout des reussites. Le journal est ecrit une fois, en ajout seul,
avant que le resultat soit connu ; le pointage vient ensuite le completer sans
jamais reecrire la ligne d'origine.

Le bilan compare systematiquement ce qui a ete *annonce* a ce qui a ete
*observe*. C'est la seule mesure qui compte : un pronostiqueur qui gagne 80%
de ses paris apres avoir annonce 90% se trompe, meme s'il gagne souvent.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from .pronostic import Pick, SettledPick, TipsterRecord, _settle
from .schema import Match, group_matches, normalize_team, parse_date

COLUMNS = [
    "pick_id", "emis_le", "date", "team_a", "team_b", "best_of", "event",
    "marche", "selection", "probabilite", "palier", "verdict",
    "cote", "edge", "mise", "score", "resultat",
]


def pick_id(pick: Pick) -> str:
    """Identifiant stable : deux fois le meme pronostic ne cree qu'une ligne."""
    raw = f"{pick.date}|{pick.team_a}|{pick.team_b}|{pick.market}|{pick.selection}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _row(pick: Pick, issued: dt.date) -> dict:
    return {
        "pick_id": pick_id(pick),
        "emis_le": issued.isoformat(),
        "date": pick.date.isoformat(),
        "team_a": pick.team_a,
        "team_b": pick.team_b,
        "best_of": pick.best_of,
        "event": pick.event,
        "marche": pick.market,
        "selection": pick.selection,
        "probabilite": f"{pick.probability:.6f}",
        "palier": pick.tier,
        "verdict": pick.verdict,
        "cote": "" if pick.odds is None else f"{pick.odds:.3f}",
        "edge": "" if pick.edge is None else f"{pick.edge:.6f}",
        "mise": "" if pick.stake is None else f"{pick.stake:.6f}",
        "score": "",
        "resultat": "",
    }


def load(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def record(picks: Iterable[Pick], path: str | Path, issued: dt.date | None = None) -> int:
    """Ajoute des pronostics au journal. Les doublons sont ignores.

    Un pronostic deja archive n'est jamais mis a jour : si l'analyse change
    apres coup, c'est une nouvelle ligne qu'il faudrait, pas une correction
    silencieuse de l'ancienne.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load(path)
    known = {r["pick_id"] for r in existing}
    issued = issued or dt.date.today()

    added = [_row(p, issued) for p in picks if pick_id(p) not in known]
    if added:
        _write(path, existing + added)
    return len(added)


def _index(matches: Sequence[Match]) -> dict:
    out = {}
    for m in matches:
        out[(m.date, frozenset((m.team_a, m.team_b)))] = m
    return out


def settle(journal_path: str | Path, results: Sequence[Match], window_days: int = 3) -> dict:
    """Complete les lignes en attente a partir des resultats reels.

    `window_days` absorbe les decalages de date (fuseau horaire, match reporte
    d'un jour) sans ouvrir la porte a un mauvais appariement.
    """
    path = Path(journal_path)
    rows = load(path)
    if not rows:
        return {"pointes": 0, "en_attente": 0, "introuvables": 0}

    idx = _index(results)
    pointes = introuvables = 0
    for row in rows:
        if row.get("resultat"):
            continue
        d = parse_date(row["date"])
        key_teams = frozenset((normalize_team(row["team_a"]), normalize_team(row["team_b"])))
        match = None
        for delta in range(0, window_days + 1):
            for signed in ({delta, -delta} if delta else {0}):
                match = idx.get((d + dt.timedelta(days=signed), key_teams))
                if match:
                    break
            if match:
                break
        if match is None:
            introuvables += 1
            continue

        # Le journal peut avoir enregistre les equipes dans l'autre sens.
        flipped = match.team_a != normalize_team(row["team_a"])
        wa, wb = (match.maps_b, match.maps_a) if flipped else (match.maps_a, match.maps_b)
        pick = Pick(
            date=d, team_a=row["team_a"], team_b=row["team_b"],
            best_of=int(row["best_of"] or 3), event=row.get("event", ""),
            market=row["marche"], selection=row["selection"],
            probability=float(row["probabilite"]), tier=row["palier"],
            verdict=row["verdict"],
        )
        meta = {"score": f"{wa}-{wb}"}
        won = _settle(pick, meta, 1.0 if wa > wb else 0.0)
        row["score"] = f"{wa}-{wb}"
        row["resultat"] = "gagne" if won else "perdu"
        pointes += 1

    _write(path, rows)
    return {
        "pointes": pointes,
        "en_attente": sum(1 for r in rows if not r.get("resultat")),
        "introuvables": introuvables,
    }


def scorecard(journal_path: str | Path) -> TipsterRecord | None:
    """Reconstruit un bilan a partir du journal pointe."""
    rows = [r for r in load(journal_path) if r.get("resultat")]
    if not rows:
        return None
    settled = [
        SettledPick(
            pick=Pick(
                date=parse_date(r["date"]), team_a=r["team_a"], team_b=r["team_b"],
                best_of=int(r["best_of"] or 3), event=r.get("event", ""),
                market=r["marche"], selection=r["selection"],
                probability=float(r["probabilite"]), tier=r["palier"],
                verdict=r["verdict"],
                odds=float(r["cote"]) if r.get("cote") else None,
                edge=float(r["edge"]) if r.get("edge") else None,
                stake=float(r["mise"]) if r.get("mise") else None,
            ),
            won=r["resultat"] == "gagne",
        )
        for r in rows
    ]
    return TipsterRecord(settled=settled, n_fixtures=len(rows), n_skipped=0)


def financial_summary(journal_path: str | Path) -> dict | None:
    """Bilan financier, uniquement sur les lignes ou une cote a ete enregistree.

    Sans cote prise au moment de l'annonce, aucun rendement n'est calculable.
    Reconstituer une cote apres coup produirait un chiffre inventé.
    """
    rows = [
        r for r in load(journal_path)
        if r.get("resultat") and r.get("cote")
    ]
    if not rows:
        return None
    flat_profit = 0.0
    kelly_profit = 0.0
    kelly_staked = 0.0
    for r in rows:
        odds = float(r["cote"])
        won = r["resultat"] == "gagne"
        flat_profit += (odds - 1.0) if won else -1.0
        stake = float(r["mise"]) if r.get("mise") else 0.0
        kelly_staked += stake
        kelly_profit += stake * (odds - 1.0) if won else -stake
    n = len(rows)
    return {
        "paris": n,
        "taux_reussite": sum(1 for r in rows if r["resultat"] == "gagne") / n,
        "cote_moyenne": sum(float(r["cote"]) for r in rows) / n,
        "profit_mise_fixe": flat_profit,
        "roi_mise_fixe": flat_profit / n,
        "profit_kelly": kelly_profit,
        "mise_kelly_totale": kelly_staked,
        "roi_kelly": kelly_profit / kelly_staked if kelly_staked > 1e-9 else 0.0,
    }
