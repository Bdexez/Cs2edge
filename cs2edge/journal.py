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
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

from .pronostic import Analysis, Pick, SettledPick, TipsterRecord, _settle
from . import __version__
from .csvio import read_rows
from .schema import Match, normalize_team, parse_date

COLUMNS = [
    "pick_id", "emis_le", "date", "team_a", "team_b", "best_of", "event",
    "marche", "selection", "probabilite", "palier", "verdict",
    "cote", "edge", "mise", "score", "resultat", "match_id", "emis_a",
    "donnees_jusquau", "version_modele",
    "carte_complete", "motif",
]


def pick_id(pick: Pick) -> str:
    """Identifiant stable : deux fois le meme pronostic ne cree qu'une ligne."""
    teams = sorted((normalize_team(pick.team_a), normalize_team(pick.team_b)))
    fixture = (f"id:{pick.match_id}" if pick.match_id else
               f"{pick.date}|{teams}|{pick.best_of}|{pick.event}")
    selection = (normalize_team(pick.selection) if pick.market == "vainqueur"
                 or pick.market.startswith("handicap") else pick.selection)
    raw = f"{fixture}|{pick.market}|{selection}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _row(pick: Pick, issued: dt.date, complete_card: bool = False) -> dict:
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
        "match_id": pick.match_id,
        "emis_a": dt.datetime.now(dt.timezone.utc).isoformat(),
        "donnees_jusquau": pick.data_through,
        "version_modele": __version__,
        "carte_complete": "1" if complete_card else "",
        "motif": "; ".join(pick.rationale),
    }


def load(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    # Les 17 colonnes du format initial restent requises ; les ajouts sont optionnels.
    return [row for _, row in read_rows(path, COLUMNS[:17])]


def _write(path: Path, rows: Sequence[dict]) -> None:
    # Remplacement atomique : une interruption ne tronque pas le journal.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=path.parent, delete=False) as fh:
            temporary = Path(fh.name)
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _locked(path: Path):
    """Verrou distinct du CSV, qui reste stable lors du remplacement atomique."""
    with path.with_name(path.name + ".lock").open("a+b") as fh:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            fh.write(b"\0")
            fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _pick_from_row(row: dict) -> Pick:
    return Pick(
        date=parse_date(row["date"]), team_a=row["team_a"], team_b=row["team_b"],
        best_of=int(row["best_of"] or 3), event=row.get("event", ""),
        market=row["marche"], selection=row["selection"],
        probability=float(row["probabilite"]), tier=row["palier"], verdict=row["verdict"],
        odds=float(row["cote"]) if row.get("cote") else None,
        edge=float(row["edge"]) if row.get("edge") else None,
        stake=float(row["mise"]) if row.get("mise") else None,
        match_id=row.get("match_id", ""), data_through=row.get("donnees_jusquau", ""),
    )


def record(picks: Iterable[Pick], path: str | Path, issued: dt.date | None = None,
           *, complete_card: bool = False) -> int:
    """Ajoute des pronostics au journal. Les doublons sont ignores.

    Un pronostic deja archive n'est jamais mis a jour : si l'analyse change
    apres coup, c'est une nouvelle ligne qu'il faudrait, pas une correction
    silencieuse de l'ancienne.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    issued = issued or dt.date.today()
    with _locked(path):
        existing = load(path)
        known = {pick_id(_pick_from_row(r)) for r in existing}
        added = []
        for pick in picks:
            identity = pick_id(pick)
            if identity not in known:
                added.append(_row(pick, issued, complete_card))
                known.add(identity)
        if added:
            _write(path, existing + added)
    return len(added)


def record_card(analyses: Sequence[Analysis], path: str | Path,
                issued: dt.date | None = None) -> int:
    """Archive aussi les abstentions pour mesurer la couverture de toute la carte."""
    picks = []
    for analysis in analyses:
        actionable = [p for p in analysis.picks if p.is_actionable]
        if actionable:
            picks.extend(actionable)
            continue
        fx, pred = analysis.fixture, analysis.prediction
        picks.append(Pick(
            fx.date, pred.team_a if pred else fx.team_a, pred.team_b if pred else fx.team_b,
            fx.best_of, fx.event, "abstention", "", 0.0, "", "ABSTENTION",
            rationale=[analysis.skipped or "aucun pronostic actionnable"], match_id=fx.match_id,
            data_through=pred.data_quality.get("data_through", "") if pred else "",
        ))
    return record(picks, path, issued, complete_card=True)


def _index(matches: Sequence[Match]) -> dict:
    out = {}
    for m in matches:
        if m.is_complete:
            teams = frozenset((normalize_team(m.team_a), normalize_team(m.team_b)))
            out.setdefault(teams, []).append(m)
    return out


def settle(journal_path: str | Path, results: Sequence[Match], window_days: int = 3) -> dict:
    """Complete les lignes en attente a partir des resultats reels.

    `window_days` absorbe les decalages de date (fuseau horaire, match reporte
    d'un jour) sans ouvrir la porte a un mauvais appariement.
    """
    path = Path(journal_path)
    if not path.exists():
        return {"pointes": 0, "en_attente": 0, "introuvables": 0, "ambigus": 0}
    if window_days < 0:
        raise ValueError("window_days doit etre positif ou nul")
    with _locked(path):
        return _settle_locked(path, results, window_days)


def _settle_locked(path: Path, results: Sequence[Match], window_days: int) -> dict:
    rows = load(path)
    if not rows:
        return {"pointes": 0, "en_attente": 0, "introuvables": 0, "ambigus": 0}

    idx = _index(results)
    pointes = introuvables = ambigus = 0
    for row in rows:
        if row.get("resultat") or row["verdict"] not in ("PRONOSTIC", "VALEUR"):
            continue
        d = parse_date(row["date"])
        key_teams = frozenset((normalize_team(row["team_a"]), normalize_team(row["team_b"])))
        candidates = [m for m in idx.get(key_teams, ())
                      if m.best_of == int(row["best_of"] or 3)]
        if row.get("match_id"):
            candidates = [m for m in candidates if m.match_id == row["match_id"]]
        else:
            candidates = [m for m in candidates if abs((m.date - d).days) <= window_days
                          and (not row.get("event") or m.event == row["event"])]
            if candidates:
                nearest = min(abs((m.date - d).days) for m in candidates)
                candidates = [m for m in candidates if abs((m.date - d).days) == nearest]
        if not candidates:
            introuvables += 1
            continue
        if len(candidates) != 1:
            ambigus += 1
            continue
        match = candidates[0]

        # Le journal peut avoir enregistre les equipes dans l'autre sens.
        flipped = normalize_team(match.team_a) != normalize_team(row["team_a"])
        wa, wb = (match.maps_b, match.maps_a) if flipped else (match.maps_a, match.maps_b)
        pick = Pick(
            date=d, team_a=normalize_team(row["team_a"]), team_b=normalize_team(row["team_b"]),
            best_of=int(row["best_of"] or 3), event=row.get("event", ""),
            market=row["marche"], selection=(normalize_team(row["selection"])
                if row["marche"] == "vainqueur" or row["marche"].startswith("handicap")
                else row["selection"]),
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
        "en_attente": sum(1 for r in rows if not r.get("resultat")
                          and r["verdict"] in ("PRONOSTIC", "VALEUR")),
        "introuvables": introuvables,
        "ambigus": ambigus,
    }


def scorecard(journal_path: str | Path) -> TipsterRecord | None:
    """Reconstruit un bilan a partir du journal pointe."""
    all_rows = load(journal_path)
    rows = [r for r in all_rows if r.get("resultat")
            and r["verdict"] in ("PRONOSTIC", "VALEUR")]
    if (not rows and not any(r["verdict"] == "ABSTENTION" for r in all_rows)
            and not any(r.get("carte_complete") == "1" for r in all_rows)):
        return None
    settled = [
        SettledPick(
            pick=_pick_from_row(r),
            won=r["resultat"] == "gagne",
        )
        for r in rows
    ]
    fixtures = {_pick_from_row(r).fixture_key for r in all_rows}
    acted = {_pick_from_row(r).fixture_key for r in all_rows
             if r["verdict"] in ("PRONOSTIC", "VALEUR")}
    return TipsterRecord(
        settled=settled, n_fixtures=len(fixtures), n_skipped=len(fixtures - acted),
        coverage_known=all(r.get("carte_complete") == "1" for r in all_rows),
        n_pending=sum(1 for r in all_rows if r["verdict"] in ("PRONOSTIC", "VALEUR")
                      and not r.get("resultat")),
    )


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
