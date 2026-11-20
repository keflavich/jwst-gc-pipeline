"""Independent, local-folder MIRI F770W workflow: discover, prepare, refit, image."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from astropy.io import fits

EXPOSURE = re.compile(r"^(jw(?P<program>\d{5})(?P<obs>\d{3})\d{3}_[a-z0-9]+_\d+_mirimage)(?:_[a-z0-9]+)?_(?P<kind>uncal|rate|cal|crf)\.fits$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_config(filename):
    filename = Path(filename).resolve()
    with filename.open("rb") as handle:
        cfg = tomllib.load(handle)
    unknown = set(cfg) - {"output_dir", "roots", "fields", "qpm", "detector1", "image2", "image3", "calibration"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if not cfg.get("output_dir") or not cfg.get("fields"):
        raise ValueError("Configuration requires output_dir and [[fields]]")

    def path(value):
        p = Path(value).expanduser()
        return str((filename.parent / p).resolve())

    cfg["output_dir"] = path(cfg["output_dir"])
    cfg["roots"] = [path(v) for v in cfg.get("roots", [])]
    names = set()
    for field in cfg["fields"]:
        unknown = set(field) - {"name", "proposal", "observations", "roots", "reference", "starless", "qpm", "train"}
        if unknown:
            raise ValueError(f"Unknown field keys: {sorted(unknown)}")
        name = field.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError(f"Invalid or duplicate field name: {name!r}")
        names.add(name)
        field["proposal"] = f"{int(field['proposal']):05d}"
        if not field.get("observations"):
            raise ValueError(f"{name}: specify observations to define the field")
        field["observations"] = [f"{int(v):03d}" for v in field["observations"]]
        if any(not re.fullmatch(r"\d{3}", v) for v in field["observations"]):
            raise ValueError(f"{name}: invalid observation number")
        field["roots"] = [path(v) for v in field.get("roots", cfg["roots"])]
        if not field["roots"]:
            raise ValueError(f"{name}: supply input roots")
        for key in ("reference", "starless"):
            if field.get(key):
                field[key] = path(field[key])
        for options in (cfg.get("qpm", {}), field.get("qpm", {})):
            for key in ("calibration_dir", "science_mask"):
                if options.get(key):
                    options[key] = path(options[key])
    calibration = cfg.get("calibration", {})
    for key in ("output_dir", "seed_dir"):
        if calibration.get(key):
            calibration[key] = path(calibration[key])
    return cfg


def discover(field, output_dir):
    """Pair exact exposure stems across folders; refuse conflicting duplicate products."""
    products, ignored = {}, []
    seen = set()
    output_dir = Path(output_dir).resolve()
    for root in field["roots"]:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Input root does not exist: {root}")
        for path in sorted(root.rglob("jw*_mirimage*.fits")):
            path = path.resolve()
            if path in seen or path == output_dir or output_dir in path.parents:
                continue
            seen.add(path)
            m = EXPOSURE.fullmatch(path.name)
            if not m or m["program"] != field["proposal"] or m["obs"] not in field["observations"]:
                continue
            h = fits.getheader(path)
            if h.get("FILTER") != "F770W" or h.get("SUBARRAY") != "FULL":
                ignored.append({"path": str(path), "reason": "requires F770W FULL"})
                continue
            if h.get("INSTRUME") != "MIRI" or h.get("EXP_TYPE") != "MIR_IMAGE":
                ignored.append({"path": str(path), "reason": "requires MIRI MIR_IMAGE"})
                continue
            kind = m["kind"]
            if kind == "crf":
                ignored.append({"path": str(path), "reason": "calibration requires a cal file before background subtraction"})
                continue
            if kind == "cal" and h.get("S_BKDSUB", "").upper() == "COMPLETE":
                ignored.append({"path": str(path), "reason": "cal has background subtraction applied"})
                continue
            key = (m[1], kind)
            if key in products and sha256(products[key]) != sha256(path):
                raise ValueError(f"Conflicting {kind} files for {m[1]}: {products[key]} and {path}. Narrow the field roots.")
            products.setdefault(key, str(path))
    rows = []
    for stem in sorted({s for s, _ in products}):
        row = {kind: products.get((stem, kind)) for kind in ("uncal", "rate", "cal")}
        if not row["rate"] and not row["uncal"]:
            raise ValueError(f"{stem}: a rate or uncal file is required for the panes")
        row.update(stem=stem, observation=stem[7:10])
        rows.append(row)
    if not rows:
        raise ValueError(f"{field['name']}: no usable F770W FULL exposures found")
    missing = set(field["observations"]) - {r["observation"] for r in rows}
    if missing:
        raise ValueError(f"{field['name']}: no usable exposures for observations {sorted(missing)}")
    return {"name": field["name"], "exposures": rows, "ignored": ignored}


def plan(cfg):
    fields = [discover(f, cfg["output_dir"]) for f in cfg["fields"]]
    # The same exposure may be deliberately included in different spatial products.
    return {"schema": 1, "config": cfg, "fields": fields}


def _fingerprint(manifest):
    sources = {}
    for field in manifest["fields"]:
        for row in field["exposures"]:
            for key in ("uncal", "rate", "cal"):
                if row.get(key):
                    p = Path(row[key])
                    sources[str(p)] = [p.stat().st_size, p.stat().st_mtime_ns]
    for field in manifest["config"]["fields"]:
        for key in ("reference", "starless"):
            if field.get(key):
                p = Path(field[key])
                sources[str(p)] = [p.stat().st_size, p.stat().st_mtime_ns]
    encoded = json.dumps([manifest, sources], sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def gather(cfg):
    manifest = plan(cfg)
    manifest["fingerprint"] = _fingerprint(manifest)
    path = Path(cfg["output_dir"]) / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Inputs or settings differ from this run. Choose a new output_dir.")
        return existing
    if path.parent.exists() and any(path.parent.iterdir()):
        raise ValueError(f"Run directory is not empty: {path.parent}")
    write_json(path, manifest)
    return manifest


def _stage(kind, source, directory, settings):
    from jwst.pipeline import Detector1Pipeline, Image2Pipeline
    pipeline = {"rate": Detector1Pipeline, "cal": Image2Pipeline}[kind]
    settings = dict(settings)
    if set(settings) - {"steps"}:
        raise ValueError("detector1 and image2 accept a steps table")
    if kind == "cal":
        steps = {k: dict(v) for k, v in settings.get("steps", {}).items()}
        steps.setdefault("resample", {})["skip"] = True
        settings["steps"] = steps
    result = pipeline.call(str(source), output_dir=str(directory), save_results=True, **settings)
    if hasattr(result, "close"):
        result.close()
    stem = EXPOSURE.fullmatch(Path(source).name)[1]
    path = directory / f"{stem}_{kind}.fits"
    if not path.is_file():
        raise RuntimeError(f"JWST stage did not produce {path}")
    return path


def _copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_suffix(".fits.partial")
        shutil.copy2(source, temp)
        temp.replace(target)
    elif sha256(source) != sha256(target):
        raise ValueError(f"Staged input differs from source: {target}")
    return target


def _reference(field, cals, directory, settings):
    if field.get("reference"):
        if not Path(field["reference"]).is_file():
            raise FileNotFoundError(field["reference"])
        return Path(field["reference"])
    from jwst.associations.asn_from_list import asn_from_list
    from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
    from jwst.pipeline import Image3Pipeline
    directory.mkdir(parents=True, exist_ok=True)
    name = field["name"] + "_f770w"
    asn = asn_from_list([str(p) for p in cals], rule=DMS_Level3_Base, product_name=name)
    asn_path = directory / "association.json"
    asn_path.write_text(asn.dump()[1])
    settings = dict(settings)
    if set(settings) - {"steps", "in_memory"}:
        raise ValueError("image3 accepts steps and in_memory")
    steps = {k: dict(v) for k, v in settings.pop("steps", {}).items()}
    steps.setdefault("source_catalog", {}).setdefault("skip", True)
    result = Image3Pipeline.call(str(asn_path), output_dir=str(directory), save_results=True,
                                 steps=steps, **settings)
    if hasattr(result, "close"):
        result.close()
    path = directory / f"{name}_i2d.fits"
    if not path.is_file():
        raise RuntimeError(f"Image3 did not produce {path}")
    return path


def prepare(manifest):
    cfg = manifest["config"]
    out = Path(cfg["output_dir"])
    receipt = out / "prepared.json"
    if receipt.exists():
        prepared = json.loads(receipt.read_text())
        for field in prepared["fields"]:
            for p in [field["reference"]] + [r[k] for r in field["exposures"] for k in ("cal", "rate")]:
                if not Path(p).is_file():
                    raise FileNotFoundError(p)
        return prepared
    fields = []
    for spec, gathered in zip(cfg["fields"], manifest["fields"]):
        folder = out / spec["name"]
        rows = []
        for row in gathered["exposures"]:
            staged = dict(row)
            rate_dir, cal_dir = folder / "rate", folder / "cal"
            rate_dir.mkdir(parents=True, exist_ok=True)
            cal_dir.mkdir(parents=True, exist_ok=True)
            rate = _copy(row["rate"], rate_dir / f"{row['stem']}_rate.fits") if row["rate"] else _stage(
                "rate", row["uncal"], rate_dir, cfg.get("detector1", {}))
            cal = _copy(row["cal"], cal_dir / f"{row['stem']}_cal.fits") if row["cal"] else _stage(
                "cal", rate, cal_dir, cfg.get("image2", {}))
            staged.update(rate=str(rate), cal=str(cal))
            rows.append(staged)
        # Image3 receives separate copies; pane photometry always reads pre-skymatch cal values.
        imager_inputs = [_copy(r["cal"], folder / "imager_inputs" / Path(r["cal"]).name) for r in rows]
        reference = _reference(spec, imager_inputs, folder / "imager", cfg.get("image3", {}))
        fields.append({**spec, "reference": str(reference), "exposures": rows})
    prepared = {"fingerprint": manifest["fingerprint"], "config": cfg, "fields": fields}
    write_json(receipt, prepared)
    return prepared


def image_fields(prepared):
    from . import calib, pipeline
    cfg = prepared["config"]
    products = []
    for field in prepared["fields"]:
        settings = {**cfg.get("qpm", {}), **field.get("qpm", {})}
        unknown = set(settings) - set(pipeline.DEFAULTS)
        reserved = set(settings) & {"sci", "rate_dir", "ref", "out_dir", "name", "inputs"}
        if unknown or reserved:
            raise ValueError(f"Invalid qpm settings: {sorted(unknown | reserved)}")
        folder = Path(cfg["output_dir"]) / field["name"] / "qpm"
        settings = {"grid": "footprint", "register": True, **settings,
                    "sci": [r["cal"] for r in field["exposures"]],
                    "rate_dir": str(Path(field["exposures"][0]["rate"]).parent),
                    "ref": field["reference"], "out_dir": str(folder), "name": field["name"] + "_f770w"}
        ref = calib.refdata(calibration_dir=settings.get("calibration_dir"))
        completion = folder / "complete.json"
        if completion.exists():
            report = json.loads(completion.read_text())
            if report["calibration_checksum"] != ref.checksum:
                raise ValueError("Calibration differs from this image run. Choose a new output_dir.")
            if not Path(report["mosaic"]).is_file():
                raise FileNotFoundError(report["mosaic"])
            products.append(report)
            continue
        rep = pipeline.run(settings, "external")
        if rep["skipped"] or rep["n_exposures"] != len(field["exposures"]):
            raise RuntimeError(f"{field['name']}: incomplete pane coverage; inspect the exposure report")
        if not rep["science_identity"]["bit_identical"]:
            raise RuntimeError(f"{field['name']}: reference science identity failed")
        counts = rep["pixels"]["counts"]
        if counts["pane_only"] + counts["pane_in_science"] == 0:
            raise RuntimeError(f"{field['name']}: no pane pixels in the final image; inspect coverage")
        report = {"field": field["name"], "mosaic": str(folder / rep["files"]["mosaic"]),
                  "report": str(folder / (settings["name"] + "_qpm_report.json")),
                  "calibration_checksum": ref.checksum, "n_exposures": rep["n_exposures"]}
        write_json(completion, report)
        products.append(report)
    write_json(Path(cfg["output_dir"]) / "products.json", products)
    return products


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "gather", "prepare", "calibrate", "image", "run"))
    parser.add_argument("--config", required=True, help="TOML workflow configuration")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.command == "plan":
            print(json.dumps(plan(cfg), indent=2))
            return
        manifest = gather(cfg)
        if args.command == "gather":
            print(Path(cfg["output_dir"]) / "manifest.json")
            return
        prepared = prepare(manifest)
        if args.command == "prepare":
            print(Path(cfg["output_dir"]) / "prepared.json")
        elif args.command == "calibrate":
            from .refit import refit
            print(refit(prepared))
        else:
            for product in image_fields(prepared):
                print(product["mosaic"])
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"gc-miri-qpm: {exc}\n")


if __name__ == "__main__":
    main()
