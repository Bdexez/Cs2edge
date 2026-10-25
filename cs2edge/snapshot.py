"""Instantane autonome de l'analyseur : historique, configuration et deux modeles.

Les composants GBDT utilisent pickle, comme MatchPredictor : charger seulement
des artefacts de confiance. Les sommes de controle detectent les alterations,
sans authentifier leur auteur.
"""

from __future__ import annotations

from dataclasses import asdict
import datetime as dt
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from . import __version__
from .features import FeatureExtractor
from .model import MatchPredictor
from .ratings import EloConfig
from .schema import ACTIVE_DUTY, MapResult, parse_date
from .validation import finite_number

_REQUIRED = {"history.json", "match.json", "map.json"}
_ALLOWED = _REQUIRED | {"match.gbdt.pkl", "map.gbdt.pkl"}
_MODEL_OPTIONS = ("l2", "use_gbdt", "gbdt_weight", "calibration_frac", "recency_halflife")


def _schemas() -> dict:
    return dict(match=list(FeatureExtractor.MATCH_FEATURES),
                map=list(FeatureExtractor.MAP_FEATURES), pool=list(ACTIVE_DUTY))


def _sklearn_version():
    try:
        return metadata.version("scikit-learn")
    except metadata.PackageNotFoundError:
        return None


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def save_analyzer(an, path: str | Path) -> None:
    if an.match_model is None or an.map_model is None or an.fx is None:
        raise RuntimeError("entrainer les deux modeles avant de sauvegarder l'analyseur")
    history = []
    for result in an.maps:
        row = asdict(result)
        row["date"] = result.date.isoformat()
        history.append(row)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, model in (("match", an.match_model), ("map", an.map_model)):
            model.save(root / f"{name}.json")
        files = {p.name: p.read_bytes() for p in root.iterdir()}
    files["history.json"] = _json_bytes(history)
    manifest = dict(
        format="cs2edge.analyzer", version=1, feature_pipeline=1, schemas=_schemas(),
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(), package_version=__version__,
        sklearn_version=_sklearn_version() if an.match_model.gbdt is not None else None,
        elo=an.fx.elo.cfg.to_dict(), warmup_maps=an.warmup_maps,
        momentum=an.momentum, momentum_auto=an._momentum_auto,
        training_options={name: getattr(an.match_model, name) for name in _MODEL_OPTIONS},
        sha256={name: hashlib.sha256(blob).hexdigest() for name, blob in files.items()},
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as fh:
            temporary = Path(fh.name)
            with ZipFile(fh, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", _json_bytes(manifest))
                for name, blob in files.items():
                    archive.writestr(name, blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_analyzer(path: str | Path):
    from .predict import Analyzer

    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            files = set(names) - {"manifest.json"}
            if (len(names) != len(set(names)) or "manifest.json" not in names
                    or not _REQUIRED <= files or not files <= _ALLOWED):
                raise ValueError("contenu d'instantane incomplet ou inconnu")
            manifest = json.loads(archive.read("manifest.json"))
            if (manifest.get("format") != "cs2edge.analyzer" or manifest.get("version") != 1
                    or manifest.get("feature_pipeline") != 1 or manifest.get("schemas") != _schemas()):
                raise ValueError("version ou schema d'instantane incompatible")
            if set(manifest.get("sha256", {})) != files:
                raise ValueError("inventaire d'instantane incoherent")
            blobs = {name: archive.read(name) for name in files}
        for name, blob in blobs.items():
            if hashlib.sha256(blob).hexdigest() != manifest["sha256"][name]:
                raise ValueError(f"somme de controle invalide : {name}")
        if any(name.endswith(".pkl") for name in blobs):
            installed = _sklearn_version()
            if installed is None or installed != manifest.get("sklearn_version"):
                raise ValueError("cet instantane GBDT requiert la meme version de scikit-learn "
                                 f"({manifest.get('sklearn_version')})")
        warmup = manifest["warmup_maps"]
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("warmup_maps doit etre un entier positif ou nul")
        finite_number("momentum", abs(manifest["momentum"]))
        if not isinstance(manifest["momentum_auto"], bool):
            raise ValueError("momentum_auto doit etre un booleen")
        options = manifest["training_options"]
        if set(options) != set(_MODEL_OPTIONS):
            raise ValueError("configuration d'entrainement incompatible")
        MatchPredictor(**options)
        history = []
        for row in json.loads(blobs["history.json"]):
            row["date"] = parse_date(row["date"])
            row["roster_a"], row["roster_b"] = tuple(row["roster_a"]), tuple(row["roster_b"])
            history.append(MapResult(**row))
        an = Analyzer(EloConfig.from_dict(manifest["elo"]), manifest["warmup_maps"],
                      manifest["momentum"]).load_maps(history)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in files - {"history.json"}:
                (root / name).write_bytes(blobs[name])
            an.match_model = MatchPredictor.load(root / "match.json")
            an.map_model = MatchPredictor.load(root / "map.json")
        for name, model, dataset in (("match", an.match_model, an.match_ds),
                                     ("map", an.map_model, an.map_ds)):
            if (model.feature_names != dataset.feature_names or model.trained_on != len(dataset)
                    or model.last_date != max(dataset.dates).isoformat()):
                raise ValueError(f"modele {name} incompatible avec l'historique de l'instantane")
        if an.map_model.gbdt is not None:
            raise ValueError("le modele map doit utiliser la regression logistique seule")
        an._training_options = options
        for model in (an.match_model, an.map_model):
            for name, value in options.items():
                setattr(model, name, value)
        an.map_model.use_gbdt = False
        an._momentum_auto = manifest["momentum_auto"]
        return an
    except (BadZipFile, KeyError, TypeError, AttributeError, UnicodeError, EOFError) as exc:
        raise ValueError(f"instantane illisible ou incomplet : {exc}") from exc
