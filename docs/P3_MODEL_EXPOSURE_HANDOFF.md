# P3 model exposure handoff

P2 supplies a genuine model run under `outputs/<run_id>/` with:

- `observed_water.tif`: one band, georeferenced, classes `0` inferred non-water, `1` inferred water, `255` invalid/no observation.
- `observed_metadata.json`: model, acquisition, inference, class mapping, threshold and output-grid provenance.

After P2 places those files in the run directory, P3/P5 should run:

```bash
python scripts/exposure.py \
  --mask outputs/<run_id>/observed_water.tif \
  --observed-metadata outputs/<run_id>/observed_metadata.json \
  --aoi config/aois/aoi_core.geojson \
  --buildings "$PRAVAH_DATA_ROOT/processed/osm/thematic/buildings.gpkg" \
  --roads "$PRAVAH_DATA_ROOT/processed/osm/thematic/roads.gpkg" \
  --bridges "$PRAVAH_DATA_ROOT/processed/osm/thematic/bridges.gpkg" \
  --settlements "$PRAVAH_DATA_ROOT/processed/osm/thematic/settlements.gpkg" \
  --schools "$PRAVAH_DATA_ROOT/processed/osm/thematic/schools.gpkg" \
  --healthcare "$PRAVAH_DATA_ROOT/processed/osm/thematic/healthcare.gpkg" \
  --emergency-facilities "$PRAVAH_DATA_ROOT/processed/osm/thematic/emergency_facilities.gpkg" \
  --osm-date 2026-01-01 \
  --out outputs/<run_id>/exposure_summary.json
```

The command validates the raster CRS, affine transform, one-band shape,
nodata convention, class values and AOI overlap. It excludes class `255` from
all coverage denominators, accepts a valid all-zero water scene, and rejects a
no-observation-only scene. It writes `exposure.geojson` and
`exposure_metadata.json` beside the summary unless alternate paths are given.

All reported intersections are potential mapped exposure. They do not establish
damage, closure, bridge failure or loss of service. The provisional bbox and
OSM snapshot must remain labelled as such until P1 supplies an authoritative
AOI and provenance.
