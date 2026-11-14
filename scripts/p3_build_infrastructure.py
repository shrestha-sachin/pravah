#!/usr/bin/env python3
"""Derive provisional bbox-clipped thematic OSM files; not event damage.

Reconstructed from the P3-local algorithm reported as successful. Validate before production.
Use --root /mnt/c/PRAVAH_DATA (default via PRAVAH_DATA_ROOT), never add data to git.
"""
import argparse
import json
import os
import re
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import make_valid

BBOX = (85.10, 27.85, 85.70, 28.35)
TAG_PATTERN = re.compile(r'"((?:\\.|[^"\\])*)"=>"((?:\\.|[^"\\])*)"')


def clean_identifier(value):
    """Return an OSM identifier or None; never turn missing values into ``nan``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text


def source_identity(row, layer: str, row_number: int) -> tuple[str, str]:
    """Build a stable namespaced identity while retaining unidentified rows."""
    relation_id = clean_identifier(row.get("osm_id"))
    way_id = clean_identifier(row.get("osm_way_id"))
    if way_id:
        return f"osm_way:{way_id}", "osm_way"
    if relation_id:
        return f"osm_relation:{relation_id}", "osm_relation"
    return f"{layer}:row:{row_number}", "unidentified_row"


def tag(row, key):
    if key in row.index:
        value = row[key]
        if value is not None:
            s = str(value)
            if s and s.lower() not in ("nan", "none"):
                return s
    tags = row.get("other_tags")
    if isinstance(tags, str):
        return dict(TAG_PATTERN.findall(tags)).get(key)
    return None


def is_true_tag(row, key):
    value = tag(row, key)
    return bool(value and value.lower() not in ("no", "false", "0"))


def polygon_part(geom):
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        return unary_union(parts) if parts else None
    return None


def export(gdf, name, category, attr, destination, bounds):
    records, repaired, skipped, multipart = [], 0, 0, 0
    for row_number, (_, row) in enumerate(gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            skipped += 1
            continue
        was_polygon = geom.geom_type in ("Polygon", "MultiPolygon")
        if not geom.is_valid:
            geom = make_valid(geom)
            repaired += 1
        if was_polygon:
            geom = polygon_part(geom)
            if geom is None or geom.is_empty:
                skipped += 1
                continue
        geom = geom.intersection(bounds)
        if geom.is_empty:
            skipped += 1
            continue
        if was_polygon:
            geom = polygon_part(geom)
            if geom is None or geom.is_empty:
                skipped += 1
                continue
        if geom.geom_type.startswith("Multi"):
            multipart += 1
        source_id, source_id_type = source_identity(row, name, row_number)
        relation_id = clean_identifier(row.get("osm_id"))
        way_id = clean_identifier(row.get("osm_way_id"))
        name_value = clean_identifier(row.get("name")) or ""
        records.append({
            "osm_id": relation_id,
            "osm_way_id": way_id,
            "source_id": source_id,
            "source_id_type": source_id_type,
            "name": name_value,
            "category": category,
            "osm_value": tag(row, attr) or "" if attr else category,
            "geometry": geom,
        })
    output = {"features": len(records), "repaired": repaired, "skipped": skipped, "multipart_geometries": multipart}
    if records:
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        path = destination / f"{name}.gpkg"
        if path.exists():
            path.unlink()
        frame.to_file(path, layer=name, driver="GPKG")
        output["file"] = path.name
    print(f"{name}: {len(records)} features | repaired: {repaired} | skipped: {skipped}")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(os.environ.get("PRAVAH_DATA_ROOT", "/mnt/c/PRAVAH_DATA")))
    args = p.parse_args()
    source = args.root / "processed/osm"
    destination = source / "thematic"
    destination.mkdir(parents=True, exist_ok=True)
    pts = gpd.read_file(source / "osm_points.gpkg")
    lines = gpd.read_file(source / "osm_lines.gpkg")
    polys = gpd.read_file(source / "osm_multipolygons.gpkg")
    for frame in (pts, lines, polys):
        if frame.crs is None or frame.crs.to_epsg() != 4326:
            raise ValueError("Expected EPSG:4326 OSM input")
    aoi = box(*BBOX)
    jobs = [
        ("roads", lines[lines.highway.notna()], "road", "highway"),
        ("bridges", lines[lines.apply(lambda r: is_true_tag(r, "bridge"), axis=1)], "bridge", "bridge"),
        ("waterways", lines[lines.waterway.notna()], "waterway", "waterway"),
        ("buildings", polys[polys.building.notna()], "building", "building"),
        ("settlements", pts[pts.place.notna()], "settlement", "place"),
        ("schools", pts[pts.apply(lambda r: tag(r, "amenity") in ("school", "college", "university", "kindergarten"), axis=1)], "education", "amenity"),
        ("healthcare", pts[pts.apply(lambda r: tag(r, "amenity") in ("hospital", "clinic", "doctors", "health_post"), axis=1)], "healthcare", "amenity"),
        ("emergency_facilities", pts[pts.apply(lambda r: tag(r, "amenity") in ("police", "fire_station", "social_facility") or tag(r, "emergency") in ("assembly_point", "shelter"), axis=1)], "emergency", "amenity"),
        ("waterbodies", polys[polys.apply(lambda r: tag(r, "natural") == "water" or tag(r, "water") is not None, axis=1)], "waterbody", "natural"),
        ("landuse", polys[polys.landuse.notna()], "landuse", "landuse"),
    ]
    summary = {
        "source": "Geofabrik Nepal 2026-01-01 OSM snapshot",
        "study_area": "Provisional bounding box; NOT official Rasuwa district boundary",
        "bbox_epsg4326": list(BBOX),
        "crs": "EPSG:4326",
        "processing": "bbox intersection, make_valid on selected invalid polygons",
        "note": "Feature presence/overlap does not establish flood exposure or damage",
        "layers": {},
    }
    for name, frame, category, attribute in jobs:
        summary["layers"][name] = export(frame, name, category, attribute, destination, aoi)
    (destination / "infrastructure_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Generated:", destination)


if __name__ == "__main__":
    main()
