"""Coverage-aware infrastructure exposure metrics.

The functions in this module measure intersections with a model-derived water
layer. They do not infer damage, closure, failure, or loss of service.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import geopandas as gpd
from shapely import make_valid
from shapely.geometry.base import BaseGeometry

DEFAULT_PROJECTED_CRS = "EPSG:32645"
MISSING_IDENTIFIER_VALUES = {"", "nan", "none", "null", "<na>"}


def _identifier_present(values):
    """Treat null-like strings as missing, including values read from GPKG."""
    text = values.astype("string").str.strip().str.lower()
    return values.notna() & ~text.isin(MISSING_IDENTIFIER_VALUES)


def _clean_layer(layer: gpd.GeoDataFrame | None, projected_crs: str) -> gpd.GeoDataFrame | None:
    if layer is None:
        return None
    if layer.crs is None:
        raise ValueError("GIS source is missing an explicit CRS")
    frame = layer.copy()
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        return frame.to_crs(projected_crs)
    invalid = ~frame.geometry.is_valid
    if invalid.any():
        frame.loc[invalid, "geometry"] = frame.loc[invalid, "geometry"].map(make_valid)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    return frame.to_crs(projected_crs)


def _deduplicate(frame: gpd.GeoDataFrame | None, id_columns: Iterable[str]) -> tuple[gpd.GeoDataFrame | None, int, str | None]:
    if frame is None or frame.empty:
        return frame, 0, None
    identifier = next(
        (name for name in id_columns if name in frame.columns and _identifier_present(frame[name]).any()),
        None,
    )
    if identifier is None:
        return frame, 0, None
    present = _identifier_present(frame[identifier])
    duplicate = present & frame[identifier].duplicated(keep="first")
    removed = int(duplicate.sum())
    return frame.loc[~duplicate].copy(), removed, identifier


def _union(frame: gpd.GeoDataFrame | None) -> BaseGeometry | None:
    if frame is None or frame.empty:
        return None
    return frame.geometry.union_all()


def _intersection_length(frame: gpd.GeoDataFrame, target: BaseGeometry) -> float:
    return float(frame.geometry.intersection(target).length.sum())


def _intersection_area(frame: gpd.GeoDataFrame, target: BaseGeometry) -> float:
    return float(frame.geometry.intersection(target).area.sum())


def _category_lengths(frame: gpd.GeoDataFrame, target: BaseGeometry) -> dict[str, float]:
    if frame.empty:
        return {}
    column = next((name for name in ("highway", "osm_value", "category") if name in frame.columns), None)
    if column is None:
        return {"unknown": _intersection_length(frame, target)}
    values = frame[column].fillna("unknown").astype(str)
    return {value: _intersection_length(frame.loc[values == value], target) for value in sorted(values.unique())}


def _feature_count(frame: gpd.GeoDataFrame | None, target: BaseGeometry | None) -> int | None:
    if frame is None:
        return None
    if target is None or frame.empty:
        return 0
    return int(frame.geometry.intersects(target).sum())


def summarize_exposure(
    water: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    roads: gpd.GeoDataFrame | None,
    bridges: gpd.GeoDataFrame | None = None,
    projected_crs: str = DEFAULT_PROJECTED_CRS,
    *,
    coverage: gpd.GeoDataFrame | None = None,
    aoi: gpd.GeoDataFrame | None = None,
    settlements: gpd.GeoDataFrame | None = None,
    schools: gpd.GeoDataFrame | None = None,
    healthcare: gpd.GeoDataFrame | None = None,
    emergency_facilities: gpd.GeoDataFrame | None = None,
    id_columns: tuple[str, ...] = ("osm_id", "id", "fid"),
) -> dict[str, Any]:
    """Compute metric intersections against valid model observation coverage.

    Coverage is the footprint of valid mask pixels (classes 0 and 1). It is
    separate from water so a valid no-water scene reports zero water exposure
    while retaining meaningful covered denominators.
    """
    if water.crs is None:
        raise ValueError("water needs an explicit CRS")
    w = _clean_layer(water, projected_crs)
    cov = _clean_layer(coverage, projected_crs)
    boundary = _clean_layer(aoi, projected_crs)
    water_target = _union(w)
    coverage_target = _union(cov) if cov is not None else water_target
    if boundary is not None:
        aoi_target = _union(boundary)
        if aoi_target is not None:
            coverage_target = aoi_target if coverage_target is None else coverage_target.intersection(aoi_target)
            if water_target is not None:
                water_target = water_target.intersection(aoi_target)

    # Prefer the namespaced identity emitted by p3_build_infrastructure.py.
    # For older layers, osm_way_id is more specific than osm_id. Missing
    # identifiers are never treated as duplicates and remain separate rows.
    identity_columns = tuple(dict.fromkeys(("source_id", "osm_way_id", "osm_id", *id_columns)))
    b, r, br = (_clean_layer(layer, projected_crs) for layer in (buildings, roads, bridges))
    b, b_removed, b_id = _deduplicate(b, identity_columns)
    r, r_removed, r_id = _deduplicate(r, identity_columns)
    br, br_removed, br_id = _deduplicate(br, identity_columns)
    settlement_frame = _clean_layer(settlements, projected_crs)
    school_frame = _clean_layer(schools, projected_crs)
    healthcare_frame = _clean_layer(healthcare, projected_crs)
    emergency_frame = _clean_layer(emergency_facilities, projected_crs)

    result: dict[str, Any] = {
        "schema_version": "p3-exposure-v2",
        "metric_crs": projected_crs,
        "water_area_m2": float(water_target.area) if water_target is not None and not water_target.is_empty else 0.0,
        "valid_coverage_area_m2": float(coverage_target.area) if coverage_target is not None and not coverage_target.is_empty else 0.0,
        "building_footprints_intersecting": _feature_count(b, water_target),
        "building_footprints_covered": _feature_count(b, coverage_target),
        "building_footprint_overlap_area_m2": _intersection_area(b, water_target) if b is not None and water_target is not None else (0.0 if b is not None else None),
        "building_footprint_area_covered_m2": _intersection_area(b, coverage_target) if b is not None and coverage_target is not None else (0.0 if b is not None else None),
        "road_length_m_total_covered": _intersection_length(r, coverage_target) if r is not None and coverage_target is not None else (0.0 if r is not None else None),
        "road_length_m_intersecting": _intersection_length(r, water_target) if r is not None and water_target is not None else (0.0 if r is not None else None),
        "road_length_m_intersecting_by_class": _category_lengths(r, water_target) if r is not None and water_target is not None else ({} if r is not None else None),
        "bridge_tagged_features_intersecting": _feature_count(br, water_target),
        "bridge_tagged_features_covered": _feature_count(br, coverage_target),
        "settlements_intersecting": _feature_count(settlement_frame, water_target),
        "schools_intersecting": _feature_count(school_frame, water_target),
        "healthcare_intersecting": _feature_count(healthcare_frame, water_target),
        "emergency_facilities_intersecting": _feature_count(emergency_frame, water_target),
        "deduplication": {
            "buildings_removed": b_removed, "buildings_id_field": b_id,
            "roads_removed": r_removed, "roads_id_field": r_id,
            "bridges_removed": br_removed, "bridges_id_field": br_id,
        },
        "caveats": [
            "Intersection with inferred water indicates potential mapped exposure, not observed damage or loss of service.",
            "OSM features are mapped records; coverage and completeness are not guaranteed.",
            "Bridge counts are bridge-tagged source features and may represent segments of one physical bridge.",
        ],
    }
    covered = result["building_footprints_covered"]
    intersecting = result["building_footprints_intersecting"]
    result["building_percent_of_covered_intersecting"] = (
        float(intersecting / covered * 100.0) if covered not in (None, 0) and intersecting is not None else None
    )
    if aoi is not None:
        aoi_target = _union(boundary)
        result["aoi_area_m2"] = float(aoi_target.area) if aoi_target is not None else 0.0
        result["coverage_fraction_of_aoi"] = (
            result["valid_coverage_area_m2"] / result["aoi_area_m2"] if result["aoi_area_m2"] else None
        )
    return result


def validate_model_mask(values: Any) -> dict[str, int]:
    """Validate project mask classes and return pixel counts."""
    import numpy as np

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("model water mask must be a two-dimensional single band array")
    if not np.isfinite(array).all() or not np.isin(array, (0, 1, 255)).all():
        raise ValueError("unsupported model mask classes; expected 0=no water, 1=water, 255=invalid")
    return {
        "water_pixels": int((array == 1).sum()),
        "non_water_pixels": int((array == 0).sum()),
        "invalid_pixels": int((array == 255).sum()),
        "valid_pixels": int(np.isin(array, (0, 1)).sum()),
    }
