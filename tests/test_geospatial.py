import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.geometry import LineString, Polygon

from pravah.gis.exposure import summarize_exposure, validate_mask_raster, validate_model_mask
from scripts.p3_build_infrastructure import source_identity


def test_projected_exposure_intersections():
    water = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32645")
    buildings = gpd.GeoDataFrame(
        {"osm_id": ["b1", "b2"]},
        geometry=[
            Polygon([(8, 8), (12, 8), (12, 12), (8, 12)]),
            Polygon([(15, 15), (16, 15), (16, 16), (15, 16)]),
        ],
        crs="EPSG:32645",
    )
    roads = gpd.GeoDataFrame(
        {"osm_id": ["r1"], "osm_value": ["primary"]},
        geometry=[LineString([(-5, 5), (15, 5)])],
        crs="EPSG:32645",
    )
    res = summarize_exposure(water, buildings, roads)
    assert res["water_area_m2"] == pytest.approx(100)
    assert res["building_footprints_intersecting"] == 1
    assert res["road_length_m_intersecting"] == pytest.approx(10)
    assert res["road_length_m_intersecting_by_class"]["primary"] == pytest.approx(10)
    assert res["bridge_tagged_features_intersecting"] is None


def test_coverage_denominator_and_id_deduplication():
    water = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])], crs="EPSG:32645")
    coverage = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32645")
    buildings = gpd.GeoDataFrame(
        {"osm_id": ["same", "same", "other"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ],
        crs="EPSG:32645",
    )
    result = summarize_exposure(water, buildings, None, coverage=coverage)
    assert result["building_footprints_intersecting"] == 1
    assert result["building_footprints_covered"] == 2
    assert result["building_percent_of_covered_intersecting"] == pytest.approx(50)
    assert result["deduplication"]["buildings_removed"] == 1
    assert result["building_footprint_overlap_area_m2"] == pytest.approx(1)


def test_no_water_scene_retains_valid_coverage():
    water = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32645")
    coverage = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32645"
    )
    buildings = gpd.GeoDataFrame(
        {"osm_id": ["b1"]},
        geometry=[Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])],
        crs="EPSG:32645",
    )
    result = summarize_exposure(water, buildings, None, coverage=coverage)
    assert result["water_area_m2"] == 0
    assert result["building_footprints_intersecting"] == 0
    assert result["building_footprints_covered"] == 1
    assert result["building_percent_of_covered_intersecting"] == 0


def test_missing_crs_is_rejected():
    water = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:32645")
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])])
    with pytest.raises(ValueError, match="CRS"):
        summarize_exposure(water, None, roads)


def test_invalid_water_mask_classes_are_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        validate_model_mask(np.array([[0, 1], [2, 255]], dtype=np.uint8))
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_model_mask(np.zeros((1, 2, 1), dtype=np.uint8))


def test_valid_mask_counts_include_no_observation():
    counts = validate_model_mask(np.array([[0, 1], [255, 0]], dtype=np.uint8))
    assert counts == {"water_pixels": 1, "non_water_pixels": 2, "invalid_pixels": 1, "valid_pixels": 3}


def test_missing_identifiers_are_kept_as_distinct_features():
    water = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32645"
    )
    buildings = gpd.GeoDataFrame(
        {"osm_id": [None, "nan", "b1"]},
        geometry=[
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ],
        crs="EPSG:32645",
    )
    result = summarize_exposure(water, buildings, None)
    assert result["building_footprints_intersecting"] == 3
    assert result["deduplication"]["buildings_removed"] == 0


def test_namespaced_source_id_deduplicates_only_repeated_source_features():
    water = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32645"
    )
    buildings = gpd.GeoDataFrame(
        {"source_id": ["osm_way:1", "osm_way:1", "osm_way:2", "buildings:row:3"]},
        geometry=[
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ],
        crs="EPSG:32645",
    )
    result = summarize_exposure(water, buildings, None)
    assert result["building_footprints_intersecting"] == 3
    assert result["deduplication"]["buildings_removed"] == 1


def test_source_identity_prefers_way_relation_and_row_fallbacks():
    assert source_identity({"osm_id": "99", "osm_way_id": "123"}, "buildings", 4) == (
        "osm_way:123", "osm_way"
    )
    assert source_identity({"osm_id": "99", "osm_way_id": None}, "buildings", 4) == (
        "osm_relation:99", "osm_relation"
    )
    assert source_identity({"osm_id": "nan", "osm_way_id": None}, "buildings", 4) == (
        "buildings:row:4", "unidentified_row"
    )


def _write_mask(path, values, *, nodata=255, x=300000, y=3100000):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=len(values),
        width=len(values[0]),
        count=1,
        dtype="uint8",
        crs="EPSG:32645",
        transform=from_origin(x, y, 10, 10),
        nodata=nodata,
    ) as dst:
        dst.write(np.asarray(values, dtype="uint8"), 1)


def test_mask_raster_checks_grid_nodata_and_aoi_overlap(tmp_path):
    path = tmp_path / "observed_water.tif"
    _write_mask(path, [[0, 1], [255, 0]])
    aoi = gpd.GeoDataFrame(
        geometry=[box(300000, 3099980, 300020, 3100000)], crs="EPSG:32645"
    )
    result = validate_mask_raster(path, aoi)
    assert result["counts"] == {"water_pixels": 1, "non_water_pixels": 2, "invalid_pixels": 1, "valid_pixels": 3}
    assert result["aoi"]["overlap"] is True
    assert result["resolution"] == [10.0, 10.0]


def test_mask_raster_rejects_wrong_nodata_and_no_observation(tmp_path):
    wrong_nodata = tmp_path / "wrong_nodata.tif"
    _write_mask(wrong_nodata, [[0, 1]], nodata=99)
    with pytest.raises(ValueError, match="nodata"):
        validate_mask_raster(wrong_nodata)

    no_observation = tmp_path / "no_observation.tif"
    _write_mask(no_observation, [[255, 255]])
    with pytest.raises(ValueError, match="no valid observation"):
        validate_mask_raster(no_observation)


def test_mask_raster_rejects_nonoverlapping_aoi(tmp_path):
    path = tmp_path / "observed_water.tif"
    _write_mask(path, [[0, 1]])
    aoi = gpd.GeoDataFrame(
        geometry=[box(400000, 3200000, 400020, 3200020)], crs="EPSG:32645"
    )
    with pytest.raises(ValueError, match="does not overlap"):
        validate_mask_raster(path, aoi)
