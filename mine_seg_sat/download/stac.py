from pathlib import Path

import geopandas as gpd
import requests
from pystac.item import Item
from pystac_client import Client
from shapely.geometry import MultiPolygon


def get_sentinel2_data(
    client: Client,
    aoi: dict,
    start_date: str,
    end_date: str,
    cloud_cover: float = 1,
    max_items: int = 500,
):
    query = client.search(
        collections=["sentinel-2-l2a"],
        datetime=f"{start_date}T00:00:00.000000Z/{end_date}T00:00:00.000000Z",  # 2023-07-10T00:00:00.000000Z/2023-07-20T00:00:00.000000Z
        intersects=aoi,
        query={"eo:cloud_cover": {"lte": cloud_cover}},
        sortby=[
            {"field": "properties.eo:cloud_cover", "direction": "asc"},
        ],
        limit=max_items,  # This is the number of items to be returned per page
        max_items=max_items,  # This is number of items to page over
    )

    items = list(query.items())
    if len(items) == 0:
        raise Exception(
            "No items found, try enlarging search area or increasing cloud cover threshold."
        )
    print(f"Found: {len(items):d} tiles.")

    # Convert STAC items into a GeoJSON FeatureCollection
    stac_json = query.item_collection_as_dict()
    gdf = gpd.GeoDataFrame.from_features(stac_json, crs="EPSG:4326")

    return items, gdf


def remove_small_tiles(
    gdf: gpd.GeoDataFrame, min_area: float = 1e6, reproject: bool = True
):
    """
    Given a GeoDataFrame as input, remove all geometries that are less than the specified area.

    Returns:
        gdf: A GeoDataFrame with geometries removed.
    """
    gdf_projected = gdf.copy()
    if reproject:
        lcc_crs = "+proj=lcc +lat_1=40 +lat_2=65 +lon_0=10 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        gdf_projected = gdf.to_crs(lcc_crs)

    gdf_projected["area_km"] = gdf_projected["geometry"].area / 1000
    gdf["proj_area"] = gdf_projected["area_km"]
    gdf = gdf.loc[gdf["proj_area"] > min_area]

    return gdf


def add_geometries_iteratively(
    gdf: gpd.GeoDataFrame, intersection_threshold: float = 0.95, debug: bool = False
) -> tuple[MultiPolygon, gpd.GeoDataFrame]:
    """
    Given a GeoDataFrame as input, construct an area iteratively by adding all of the geometries together.

    Returns:
        merged_geometry: A shapely Polygon or MultiPolygon object representing the merged geometry.
        selected_geometries: A list of GeoDataFrame rows that were selected to be merged.
    """
    assert intersection_threshold < 1, "The intersection threshold must be less than 1."
    # Initialize an empty geometry
    merged_geometry = None
    selected_geometries = []

    # Iterate over each row in the GeoDataFrame
    for idx, row in gdf.iterrows():
        geometry = row["geometry"]
        intersected = False  # Flag to track if the row has intersected geometry

        # Check if the current geometry is a MultiPolygon
        if isinstance(geometry, MultiPolygon):
            # Iterate over each polygon in the MultiPolygon
            for polygon in geometry.geoms:
                if merged_geometry is None:
                    merged_geometry = polygon
                else:
                    if (
                        merged_geometry.intersection(polygon).area / polygon.area
                    ) > intersection_threshold:
                        intersected = True
                    else:
                        merged_geometry = merged_geometry.union(polygon)
        else:
            # If the current geometry is not a MultiPolygon
            if merged_geometry is None:
                merged_geometry = geometry
            else:
                if (
                    merged_geometry.intersection(geometry).area / geometry.area
                ) > intersection_threshold:
                    intersected = True
                else:
                    merged_geometry = merged_geometry.union(geometry)

        # Print a message if the row has no non-intersecting area
        if intersected:
            if debug:
                print(
                    f"Row {idx} has no area that does not already intersect with the merged geometry."
                )
        else:
            selected_geometries.append(row)

    return (merged_geometry, gpd.GeoDataFrame(selected_geometries, crs=gdf.crs))


def download_file(href: str, outpath: Path):
    """
    Download a file from the Sentinel Hub API.
    """
    with requests.get(href, stream=True) as r:
        r.raise_for_status()
        with open(outpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_files_for_item(
    item: Item, asset_dict: dict[str, str], outpath: Path
) -> bool:
    """
    Save all files of interest for a given item.

    If one file fails to download return False, otherwise return True.
    """
    if not outpath.exists():
        outpath.mkdir(parents=True, exist_ok=True)

    for key, value in asset_dict.items():
        if key in item.assets:
            if key in ["productinfo", "metadata"]:
                file_outpath = outpath / f"{value}.json"
            file_outpath = outpath / f"{value}.tif"
            if not file_outpath.exists():
                try:
                    download_file(item.assets[key].href, file_outpath)
                except requests.ConnectionError:
                    print(
                        f"Failed to download {item.assets[key].href} for item {item.id}"
                    )
                    return False
                except requests.exceptions.ReadTimeout:
                    print(
                        f"Experienced a read timeout for {item.assets[key].href} for item {item.id}"
                    )
                    return False
            else:
                print(f"Skipping {item.assets[key].href} as it already exists.")

    return True


def download_all_sentinel2_items(
    gdf: gpd.GeoDataFrame,
    items: list[Item],
    asset_dict: dict[str, str],
    base_path: Path,
):
    """
    Download all files for all items in a GeoDataFrame.
    """
    assert len(gdf) == len(items), "GeoDataFrame and items list must be of same length"
    gdf["downloaded"] = False
    for item in items:
        downloaded = download_files_for_item(item, asset_dict, base_path)
        gdf.loc[
            gdf["s2:granule_id"] == item.properties["s2:granule_id"], "downloaded"
        ] = downloaded

        print(f"Downloaded all files for item {item.id}")

    return gdf
