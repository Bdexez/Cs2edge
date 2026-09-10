"""Modele de donnees et normalisation des noms d'equipes."""

from __future__ import annotations

import csv
import datetime as dt
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

ACTIVE_DUTY = (
    "ancient",
    "anubis",
    "dust2",
    "inferno",
    "mirage",
    "nuke",
    "train",
)

# Tiers d'evenement -> poids de fiabilite du resultat (K-factor Elo).
TIER_WEIGHT = {"s": 1.30, "a": 1.10, "b": 1.00, "c": 0.80, "d": 0.60}

_ALIASES = {
    "natus vincere": "navi",
    "na'vi": "navi",
    "team vitality": "vitality",
    "team spirit": "spirit",
    "team falcons": "falcons",
    "team liquid": "liquid",
    "ninjas in pyjamas": "nip",
    "faze clan": "faze",
    "g2 esports": "g2",
    "mouz": "mousesports",
    "mouz nxt": "mouz nxt",
    "complexity gaming": "complexity",
    "the mongolz": "mongolz",
    "virtus.pro": "virtuspro",
    "virtus pro": "virtuspro",
    "eternal fire": "eternalfire",
    "9z team": "9z",
}

_MAP_ALIASES = {
    "de_dust2": "dust2",
    "d2": "dust2",
    "de_mirage": "mirage",
    "de_inferno": "inferno",
    "de_nuke": "nuke",
    "de_ancient": "ancient",
    "de_anubis": "anubis",
    "de_train": "train",
    "de_overpass": "overpass",
    "de_vertigo": "vertigo",
}


def normalize_team(name: str) -> str:
    """Canonicalise un nom d'equipe (casse, accents, ponctuation, alias)."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    if s in _ALIASES:
        return _ALIASES[s]
    s = re.sub(r"\b(esports?|gaming|team|clan|gg)\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return _ALIASES.get(s, s)


def normalize_map(name: str) -> str:
    s = str(name).strip().lower().replace(" ", "")
    return _MAP_ALIASES.get(s, s)


def parse_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(s[: len(fmt) + 2].strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"date illisible: {value!r}")


@dataclass(frozen=True)
class MapResult:
    """Une map jouee. C'est l'unite atomique du dataset."""

    match_id: str
    date: dt.date
    team_a: str
    team_b: str
    map_name: str
    rounds_a: int
    rounds_b: int
    best_of: int = 3
    event: str = ""
    tier: str = "b"
    lan: bool = False
    map_index: int = 0
    roster_a: tuple[str, ...] = ()
    roster_b: tuple[str, ...] = ()

    @property
    def winner(self) -> str:
        return self.team_a if self.rounds_a > self.rounds_b else self.team_b

    @property
    def a_won(self) -> int:
        return int(self.rounds_a > self.rounds_b)

    @property
    def round_diff(self) -> int:
        return self.rounds_a - self.rounds_b

    @property
    def tier_weight(self) -> float:
        return TIER_WEIGHT.get(self.tier.lower(), 1.0)


@dataclass
class Match:
    """Un match = une ou plusieurs maps entre les deux memes equipes."""

    match_id: str
    date: dt.date
    team_a: str
    team_b: str
    best_of: int
    event: str
    tier: str
    lan: bool
    maps: list[MapResult] = field(default_factory=list)

    @property
    def maps_a(self) -> int:
        return sum(m.a_won for m in self.maps)

    @property
    def maps_b(self) -> int:
        return len(self.maps) - self.maps_a

    @property
    def a_won(self) -> int:
        return int(self.maps_a > self.maps_b)

    @property
    def winner(self) -> str:
        return self.team_a if self.a_won else self.team_b


def _to_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "lan", "oui"}


def _roster(v) -> tuple[str, ...]:
    if not v:
        return ()
    return tuple(sorted(p.strip().lower() for p in str(v).split(",") if p.strip()))


def read_csv(path: str) -> list[MapResult]:
    """Charge un CSV de resultats au niveau map.

    Colonnes requises : match_id,date,team_a,team_b,map,rounds_a,rounds_b
    Optionnelles      : best_of,event,tier,lan,map_index,roster_a,roster_b
    """
    rows: list[MapResult] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = {"match_id", "date", "team_a", "team_b", "map", "rounds_a", "rounds_b"} - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(f"colonnes manquantes dans {path}: {sorted(missing)}")
        for i, row in enumerate(reader):
            try:
                rows.append(
                    MapResult(
                        match_id=str(row["match_id"]).strip(),
                        date=parse_date(row["date"]),
                        team_a=normalize_team(row["team_a"]),
                        team_b=normalize_team(row["team_b"]),
                        map_name=normalize_map(row["map"]),
                        rounds_a=int(float(row["rounds_a"])),
                        rounds_b=int(float(row["rounds_b"])),
                        best_of=int(float(row.get("best_of") or 3)),
                        event=(row.get("event") or "").strip(),
                        tier=(row.get("tier") or "b").strip().lower(),
                        lan=_to_bool(row.get("lan")),
                        map_index=int(float(row.get("map_index") or 0)),
                        roster_a=_roster(row.get("roster_a")),
                        roster_b=_roster(row.get("roster_b")),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"ligne {i + 2} invalide dans {path}: {exc}") from exc
    rows = [r for r in rows if r.rounds_a != r.rounds_b and r.team_a != r.team_b]
    rows.sort(key=lambda m: (m.date, m.match_id, m.map_index))
    return rows


def write_csv(path: str, maps: Sequence[MapResult]) -> None:
    cols = [
        "match_id", "date", "team_a", "team_b", "map", "rounds_a", "rounds_b",
        "best_of", "event", "tier", "lan", "map_index",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in maps:
            w.writerow([
                m.match_id, m.date.isoformat(), m.team_a, m.team_b, m.map_name,
                m.rounds_a, m.rounds_b, m.best_of, m.event, m.tier,
                int(m.lan), m.map_index,
            ])


def group_matches(maps: Iterable[MapResult]) -> list[Match]:
    """Regroupe les maps en matchs, en conservant l'ordre chronologique."""
    order: list[str] = []
    buckets: dict[str, list[MapResult]] = {}
    for m in maps:
        if m.match_id not in buckets:
            buckets[m.match_id] = []
            order.append(m.match_id)
        buckets[m.match_id].append(m)

    matches: list[Match] = []
    for mid in order:
        group = sorted(buckets[mid], key=lambda m: m.map_index)
        head = group[0]
        # Une equipe peut apparaitre en team_b sur une map : on realigne sur head.
        aligned: list[MapResult] = []
        for m in group:
            if m.team_a == head.team_a:
                aligned.append(m)
            else:
                aligned.append(
                    MapResult(
                        match_id=m.match_id, date=m.date, team_a=head.team_a,
                        team_b=head.team_b, map_name=m.map_name,
                        rounds_a=m.rounds_b, rounds_b=m.rounds_a,
                        best_of=m.best_of, event=m.event, tier=m.tier, lan=m.lan,
                        map_index=m.map_index, roster_a=m.roster_b, roster_b=m.roster_a,
                    )
                )
        matches.append(
            Match(
                match_id=mid, date=head.date, team_a=head.team_a, team_b=head.team_b,
                best_of=head.best_of, event=head.event, tier=head.tier, lan=head.lan,
                maps=aligned,
            )
        )
    matches.sort(key=lambda m: (m.date, m.match_id))
    return matches


def iter_chronological(matches: Sequence[Match]) -> Iterator[Match]:
    yield from sorted(matches, key=lambda m: (m.date, m.match_id))
