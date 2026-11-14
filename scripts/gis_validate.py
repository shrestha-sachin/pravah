#!/usr/bin/env python3
"""Validate PRAVAH GIS inputs and write a machine-readable QA report.

The validator reports provisional/local data faithfully. It never turns the
provisional bbox into an authoritative district or flood boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

THEMATIC = (
    "roads", "bridges", "waterways", "buildings", "settlements",
    "schools", "healthcare", "emergency_facilities", "waterbodies", "landuse",
)
RASTERS = {
    "dem": ("processed/terrain/rasuwa_area_dem_utm.tif", 32645),
    "slope": ("processed/terrain/rasuwa_area_slope_degrees.tif", 32645),
    "rainfall_accumulation": ("processed/rainfall/rainfall_accumulation_aug01_26.tif", 4326),
}
DATE_RE = re.compile(r"rainfall_(\d{4}-\d{2}-\d{2})\.tif$")
MISSING_IDENTIFIER_VALUES = {"", "nan", "none", "null", "<na>"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounds_dict(bounds: tuple[float, float, float, float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(("west", "south", "east", "north"), bounds)}


def _raster_stats(ds: rasterio.DatasetReader, array: np.ndarray) -> dict[str, Any]:
    nodata = ds.nodata
    invalid = ~np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        invalid |= array == nodata
    invalid |= array == -9999
    valid = array[~invalid]
    return {
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "finite_invalid_pixels": int(invalid.sum()),
        "valid_pixels": int(valid.size),
        "nodata_declared": None if nodata is None or not np.isfinite(nodata) else float(nodata),
        "sentinel_minus_9999_pixels": int((array == -9999).sum()),
    }


def _load_aoi(path: Path | None, bbox: list[float] | None) -> gpd.GeoDataFrame | None:
    if path is not None:
        frame = gpd.read_file(path)
        if frame.empty or frame.crs is None:
            raise ValueError("AOI must contain features and an explicit CRS")
        return frame
    if bbox is not None:
        west, south, east, north = bbox
        if not west < east or not south < north:
            raise ValueError("bbox must be WEST SOUTH EAST NORTH")
        return gpd.GeoDataFrame({"name": ["aoi_provisional"]}, geometry=[box(west, south, east, north)], crs="EPSG:4326")
    return None


def _present_identifier(values: Any) -> np.ndarray:
    series = values.astype("string").str.strip().str.lower()
    return (values.notna() & ~series.isin(MISSING_IDENTIFIER_VALUES)).to_numpy()


def _identifier_audit(frame: gpd.GeoDataFrame) -> dict[str, Any]:
    fields = [name for name in ("source_id", "osm_id", "osm_way_id") if name in frame.columns]
    audit: dict[str, Any] = {"fields": fields, "by_field": {}, "missing_all_identifiers": 0}
    present_masks: list[np.ndarray] = []
    for field in fields:
        present = _present_identifier(frame[field])
        values = frame.loc[present, field].astype("string")
        duplicate_rows = int(values.duplicated(keep=False).sum())
        audit["by_field"][field] = {
            "populated": int(present.sum()),
            "unique_nonempty": int(values.nunique(dropna=True)),
            "duplicate_nonempty_rows": duplicate_rows,
            "missing": int((~present).sum()),
        }
        present_masks.append(present)
    if present_masks:
        audit["missing_all_identifiers"] = int((~np.logical_or.reduce(present_masks)).sum())
    else:
        audit["missing_all_identifiers"] = int(len(frame))
    return audit


def _validate_vector(path: Path, aoi: gpd.GeoDataFrame | None) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    frame = gpd.read_file(path)
    item: dict[str, Any] = {
        "path": str(path),
        "count": int(len(frame)),
        "crs": str(frame.crs) if frame.crs else None,
        "bounds": _bounds_dict(tuple(frame.total_bounds)) if not frame.empty else None,
        "geometry_types": {str(k): int(v) for k, v in frame.geom_type.value_counts().items()},
        "multipart_geometry_count": int(frame.geom_type.astype("string").str.startswith("Multi").sum()),
        "null_geometry": int(frame.geometry.isna().sum()),
        "empty_geometry": int(frame.geometry.is_empty.sum()),
        "invalid_geometry": int((~frame.geometry.is_valid).sum()),
        "duplicate_ids": 0,
        "id_field": None,
        "identifier_audit": _identifier_audit(frame),
        "aoi_intersects": None,
    }
    identifier = next((name for name in ("source_id", "osm_way_id", "osm_id", "id", "fid") if name in frame.columns), None)
    if identifier:
        item["id_field"] = identifier
        present = _present_identifier(frame[identifier])
        item["duplicate_ids"] = int((frame.loc[present, identifier].duplicated(keep=False)).sum())
    if aoi is not None and not frame.empty:
        aoi_metric = aoi.to_crs(frame.crs)
        item["aoi_intersects"] = int(frame.geometry.intersects(aoi_metric.geometry.union_all()).sum())
    if item["crs"] is None:
        errors.append(f"{path.name}: missing CRS")
    if item["null_geometry"] or item["empty_geometry"] or item["invalid_geometry"]:
        errors.append(f"{path.name}: null, empty, or invalid geometries")
    # OSM thematic outputs can contain repeated relation/way IDs when a
    # physical feature is represented by multiple mapped geometries. Keep the
    # count explicit; exposure calculations deduplicate where required.
    return item, errors


def _validate_raster(path: Path, expected_epsg: int, aoi: gpd.GeoDataFrame | None) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    with rasterio.open(path) as ds:
        array = ds.read(1)
        item: dict[str, Any] = {
            "path": str(path),
            "crs": str(ds.crs) if ds.crs else None,
            "epsg": ds.crs.to_epsg() if ds.crs else None,
            "width": int(ds.width), "height": int(ds.height), "bands": int(ds.count),
            "dtype": ds.dtypes[0], "transform": [float(v) for v in ds.transform[:6]],
            "resolution": [float(ds.transform.a), float(abs(ds.transform.e))],
            "bounds": _bounds_dict((ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)),
            **_raster_stats(ds, array),
            "sha256": _sha256(path),
            "aoi_overlap": None,
        }
        if aoi is not None and ds.crs is not None:
            aoi_raster = aoi.to_crs(ds.crs)
            item["aoi_overlap"] = bool(box(*ds.bounds).intersects(aoi_raster.geometry.union_all()))
        if item["crs"] is None or item["epsg"] != expected_epsg:
            errors.append(f"{path.name}: expected EPSG:{expected_epsg}")
        if ds.count != 1 or ds.width <= 0 or ds.height <= 0:
            errors.append(f"{path.name}: invalid raster shape/bands")
        if item["valid_pixels"] == 0:
            errors.append(f"{path.name}: no valid pixels")
        if aoi is not None and item["aoi_overlap"] is False:
            errors.append(f"{path.name}: no overlap with AOI")
    return item, errors


def _rainfall_manifest(root: Path, aoi: gpd.GeoDataFrame | None) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    errors: list[str] = []
    entries: list[dict[str, Any]] = []
    files = sorted((root / "processed/rainfall").glob("rainfall_*.tif"))
    dates: list[date] = []
    for path in files:
        match = DATE_RE.match(path.name)
        if not match:
            continue
        observed = date.fromisoformat(match.group(1))
        dates.append(observed)
        item, item_errors = _validate_raster(path, 4326, aoi)
        item.update({"date": observed.isoformat(), "product": "CHIRPS v3 FINAL", "availability_status": "unverified_retrospective"})
        entries.append(item)
        errors.extend(item_errors)
    if dates:
        expected = list(range((max(dates) - min(dates)).days + 1))
        if len(dates) != len(expected) or len(set(dates)) != len(dates):
            errors.append("rainfall daily manifest has missing or duplicate dates")
    accumulation: dict[str, Any] = {}
    acc_path = root / RASTERS["rainfall_accumulation"][0]
    if acc_path.exists() and entries:
        with rasterio.open(acc_path) as acc:
            stack = []
            for entry in entries:
                with rasterio.open(entry["path"]) as daily:
                    if daily.shape != acc.shape or daily.transform != acc.transform or daily.crs != acc.crs:
                        errors.append(f"{Path(entry['path']).name}: not aligned with accumulation raster")
                    stack.append(daily.read(1).astype("float64"))
            expected_sum = np.sum(stack, axis=0)
            actual = acc.read(1).astype("float64")
            mismatch = int(np.sum(~np.isclose(expected_sum, actual, rtol=1e-5, atol=1e-4)))
            accumulation = {"path": str(acc_path), "daily_files": len(stack), "mismatched_pixels": mismatch}
            if mismatch:
                errors.append("rainfall accumulation does not equal daily raster sum")
    return entries, errors, accumulation


def validate(root: Path | str, aoi: gpd.GeoDataFrame | None = None, bbox: list[float] | None = None) -> dict[str, Any]:
    root = Path(root)
    if aoi is None and bbox is not None:
        aoi = _load_aoi(None, bbox)
    report: dict[str, Any] = {
        "schema_version": "p3-gis-validation-v2",
        "data_root": str(root),
        "aoi": {
            "name": "aoi_provisional_bbox" if bbox is not None else "external_aoi",
            "crs": str(aoi.crs) if aoi is not None else None,
            "bounds": _bounds_dict(tuple(aoi.to_crs("EPSG:4326").total_bounds)) if aoi is not None else None,
        },
        "sources": {
            "osm_snapshot": "2026-01-01",
            "rainfall_product": "CHIRPS v3 FINAL",
            "rainfall_availability": "unverified_retrospective",
            "terrain_product": "Copernicus GLO-30 DSM",
        },
        "source_vectors": {}, "vectors": {}, "rasters": {}, "rainfall_daily": [], "rainfall_accumulation": {},
        "warnings": [], "errors": [],
    }
    if not root.exists():
        report["errors"].append(f"data root does not exist: {root}")
        return report
    raw_multipolygons = root / "processed/osm/osm_multipolygons.gpkg"
    if raw_multipolygons.exists():
        try:
            raw = gpd.read_file(raw_multipolygons)
            report["source_vectors"]["osm_multipolygons"] = {
                "path": str(raw_multipolygons),
                "count": int(len(raw)),
                "crs": str(raw.crs) if raw.crs else None,
                "geometry_types": {str(k): int(v) for k, v in raw.geom_type.value_counts().items()},
                "invalid_geometry": int((~raw.geometry.is_valid).sum()),
                "identifier_audit": _identifier_audit(raw),
            }
            if report["source_vectors"]["osm_multipolygons"]["invalid_geometry"]:
                report["warnings"].append("osm_multipolygons source contains invalid geometries; thematic export repairs selected features")
        except Exception as exc:
            report["errors"].append(f"osm_multipolygons source: {exc}")
    for name in THEMATIC:
        path = root / "processed/osm/thematic" / f"{name}.gpkg"
        if not path.exists():
            report["errors"].append(f"missing vector {name}: {path}")
            continue
        try:
            item, errors = _validate_vector(path, aoi)
            report["vectors"][name] = item
            report["errors"].extend(errors)
            if item.get("duplicate_ids"):
                report["warnings"].append(
                    f"{name}: {item['duplicate_ids']} duplicate source IDs; counts are mapped geometries and require explicit deduplication"
                )
            missing = item["identifier_audit"]["missing_all_identifiers"]
            if missing:
                report["warnings"].append(f"{name}: {missing} features have no source identifier and remain distinct")
        except Exception as exc:
            report["errors"].append(f"{name}: {exc}")
    for name, (relative, epsg) in RASTERS.items():
        path = root / relative
        if not path.exists():
            report["errors"].append(f"missing raster {name}: {path}")
            continue
        try:
            item, errors = _validate_raster(path, epsg, aoi)
            report["rasters"][name] = item
            report["errors"].extend(errors)
        except Exception as exc:
            report["errors"].append(f"{name}: {exc}")
    try:
        daily, errors, accumulation = _rainfall_manifest(root, aoi)
        report["rainfall_daily"] = daily
        report["rainfall_accumulation"] = accumulation
        report["errors"].extend(errors)
    except Exception as exc:
        report["errors"].append(f"rainfall manifest: {exc}")
    metadata = root / "metadata"
    report["source_metadata_status"] = "present" if metadata.exists() and any(metadata.iterdir()) else "missing_or_unverified"
    report["warnings"].append("AOI is provisional unless an authoritative boundary is supplied.")
    report["warnings"].append("OSM intersection counts represent mapped features and potential exposure, not damage.")
    report["warnings"].append("CHIRPS FINAL publication availability before the historical cutoff is unverified.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PRAVAH_DATA_ROOT", r"C:\PRAVAH_DATA")))
    parser.add_argument("--aoi", type=Path, help="GeoJSON/GPKG AOI with CRS")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--out", type=Path, help="JSON report path")
    args = parser.parse_args()
    try:
        aoi = _load_aoi(args.aoi, args.bbox)
        report = validate(args.root, aoi=aoi, bbox=args.bbox)
    except Exception as exc:
        print(f"fatal validation setup error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
