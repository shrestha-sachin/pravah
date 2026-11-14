import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from pravah.gis.exposure import summarize_exposure, validate_model_mask
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
