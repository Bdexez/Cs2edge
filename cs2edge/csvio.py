"""Lecture CSV commune : ne pas masquer une colonne ou une ligne tronquee."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
from typing import Iterable, Iterator


def read_rows(path: str | Path, required: Iterable[str]) -> Iterator[tuple[int, dict[str, str]]]:
    """Produit les lignes et leur numero physique, apres controle de structure.

    Les champs vides explicites sont conserves. Une cellule absente, une cellule
    surnumeraire ou un nom de colonne duplique est une erreur, jamais une valeur
    par defaut ni un ecrasement silencieux.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, strict=True)
        try:
            fields = reader.fieldnames or []
            duplicates = sorted(name for name, count in Counter(fields).items() if count > 1)
            if duplicates:
                raise ValueError(f"colonnes dupliquees dans {path}: {duplicates}")
            missing = set(required) - set(fields)
            if missing:
                raise ValueError(f"colonnes manquantes dans {path}: {sorted(missing)}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"ligne {reader.line_num} invalide dans {path}: "
                                     f"nombre de champs different de l'entete ({len(fields)})")
                yield reader.line_num, row
        except csv.Error as exc:
            raise ValueError(f"CSV invalide dans {path}, ligne {reader.reader.line_num}: {exc}") from exc
