"""Modele de donnees et normalisation des noms d'equipes."""

from __future__ import annotations

import csv
import datetime as dt
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
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
    """Lit toute la valeur ; un horodatage conserve son jour calendaire local."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"]
    formats.extend(f"%Y-%m-%d{sep}%H:%M:%S{fraction}{zone}"
                   for sep in ("T", " ") for fraction in ("", ".%f") for zone in ("", "%z"))
    for fmt in formats:
        try:
            return dt.datetime.strptime(s, fmt).date()
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

    def __post_init__(self) -> None:
        for name in ("match_id", "team_a", "team_b", "map_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} doit etre renseigne")
        if self.team_a == self.team_b:
            raise ValueError("une equipe ne joue pas contre elle-meme")
        for name in ("rounds_a", "rounds_b", "map_index", "best_of"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} doit etre un entier positif ou nul")
        if self.best_of not in (1, 3, 5):
            raise ValueError(f"format non gere : bo{self.best_of}")
        if self.rounds_a == self.rounds_b:
            raise ValueError("une map terminee ne peut pas etre nulle")

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
        if not self.is_complete:
            raise ValueError(f"serie incomplete : {self.match_id}")
        return int(self.maps_a > self.maps_b)

    @property
    def is_complete(self) -> bool:
        need = self.best_of // 2 + 1
        return max(self.maps_a, self.maps_b) == need and min(self.maps_a, self.maps_b) < need

    @property
    def winner(self) -> str:
        return self.team_a if self.a_won else self.team_b


def _to_bool(v) -> bool:
    value = str(v).strip().lower() if v is not None else ""
    if value in {"1", "true", "yes", "y", "lan", "oui"}:
        return True
    if value in {"", "0", "false", "no", "n", "online", "non"}:
        return False
    raise ValueError(f"booleen illisible : {v!r}")


def _roster(v) -> tuple[str, ...]:
    if not v:
        return ()
    players = v.split(",") if isinstance(v, str) else v
    return tuple(sorted(str(p).strip().lower() for p in players if str(p).strip()))


def parse_integer(value, name: str) -> int:
    """Accepte 13 et 13.0, sans tronquer les fractions ni les non-finis."""
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError
        return int(number)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} doit etre un entier : {value!r}") from exc


def map_from_record(row: dict) -> MapResult:
    """Normalisation partagee par les entrees CSV et API."""
    def optional(name, default):
        value = row.get(name)
        return default if value is None or value == "" else value

    return MapResult(
        match_id=str(row.get("match_id") or "").strip(),
        date=parse_date(row["date"]),
        team_a=normalize_team(row.get("team_a") or ""),
        team_b=normalize_team(row.get("team_b") or ""),
        map_name=normalize_map(row.get("map") or row.get("map_name") or ""),
        rounds_a=parse_integer(row["rounds_a"], "rounds_a"),
        rounds_b=parse_integer(row["rounds_b"], "rounds_b"),
        best_of=parse_integer(optional("best_of", 3), "best_of"),
        event=str(row.get("event") or "").strip(),
        tier=str(row.get("tier") or "b").strip().lower(),
        lan=_to_bool(row.get("lan")),
        map_index=parse_integer(optional("map_index", 0), "map_index"),
        roster_a=_roster(row.get("roster_a")), roster_b=_roster(row.get("roster_b")),
    )


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
                rows.append(map_from_record(row))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"ligne {i + 2} invalide dans {path}: {exc}") from exc
    rows.sort(key=lambda m: (m.date, m.match_id, m.map_index))
    return rows


def write_csv(path: str, maps: Sequence[MapResult]) -> None:
    cols = [
        "match_id", "date", "team_a", "team_b", "map", "rounds_a", "rounds_b",
        "best_of", "event", "tier", "lan", "map_index", "roster_a", "roster_b",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for m in maps:
            w.writerow([
                m.match_id, m.date.isoformat(), m.team_a, m.team_b, m.map_name,
                m.rounds_a, m.rounds_b, m.best_of, m.event, m.tier,
                int(m.lan), m.map_index, ",".join(m.roster_a), ",".join(m.roster_b),
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
        indices = [m.map_index for m in group]
        if len({m.map_name for m in group}) != len(group):
            raise ValueError(f"map dupliquee dans la serie : {mid}")
        if len(set(indices)) != len(indices) and any(indices):
            raise ValueError(f"indices de map dupliques : {mid}")
        for m in group:
            if ({m.team_a, m.team_b} != {head.team_a, head.team_b}
                    or (m.date, m.best_of, m.event, m.lan, m.tier)
                    != (head.date, head.best_of, head.event, head.lan, head.tier)):
                raise ValueError(f"maps incoherentes dans le match {mid}")
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
        need = head.best_of // 2 + 1
        wa = wb = 0
        for m in aligned:
            if wa == need or wb == need:
                raise ValueError(f"map apres la fin de la serie : {mid}")
            wa += m.a_won
            wb += 1 - m.a_won
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
