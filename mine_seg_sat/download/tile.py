import typing as T
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon


def get_aoi_shape(gdf: gpd.GeoDataFrame, target_crs: str) -> MultiPolygon:
    return gdf.to_crs(target_crs).unary_union


def create_masks(gdf: Path, out_path: Path, bounding_box: Polygon, meta: dict):
    gdf = gdf[gdf.intersects(bounding_box)]
    if len(gdf) == 0:
        raise ValueError("Intersecting area must have at least one polygon.")
    geometries = [(geom, value) for geom, value in zip(gdf.geometry, gdf.label)]

    with rasterio.open(out_path, "w+", **meta) as dest:
        out_arr = dest.read(1)
        burned = rasterize(
            geometries,
            out=out_arr,
            transform=dest.transform,
            fill=0,
            default_value=0,
            dtype=rasterio.uint8,
        )
        dest.write_band(1, burned)


def generate_tiles(
    input_file: Path,
    output_dir: Path,
    band_name: str,
    window_size: int = 768,
    aoi_gdf: T.Optional[gpd.GeoDataFrame] = None,
) -> int:
    """
    Split Sentinel files into square tiles of a given window size.
    """
    all_bounds = []
    with rasterio.open(input_file.as_posix()) as src:
        height, width = src.shape

        # Calculate the number of segments in both dimensions
        num_rows = (height + window_size - 1) // window_size
        num_cols = (width + window_size - 1) // window_size
        num_processed = 0
        skipped = 0
        blank = 0
        misshapen = 0
        no_mask = 0
        aoi_gdf["label"] = 1
        aoi_gdf = aoi_gdf.to_crs(src.crs)
        aoi = aoi_gdf.unary_union

        for row in range(num_rows):
            for col in range(num_cols):
                row_start = row * window_size
                col_start = col * window_size

                # Calculate the actual segment size based on the remaining pixels
                seg_height = min(window_size, height - row_start)
                seg_width = min(window_size, width - col_start)
                seg_window = Window(col_start, row_start, seg_width, seg_height)
                # skip if window is not a perfect square or the shape of the window is not the same as the window size
                if seg_height != seg_width or seg_height != window_size:
                    skipped += 1
                    misshapen += 1
                    continue

                seg_data = src.read(window=seg_window)
                seg_profile = src.profile.copy()
                seg_transform = src.window_transform(
                    seg_window
                )  # don't use rasterio.windows.transform...

                # Create a Polygon for the window's bounds
                bound = [
                    seg_transform * (0, 0),
                    seg_transform * (seg_window.width, 0),
                    seg_transform * (seg_window.width, seg_window.height),
                    seg_transform * (0, seg_window.height),
                ]
                polygon = Polygon(bound)

                if not polygon.intersects(aoi):
                    no_mask += 1
                    continue

                all_bounds.append(polygon)
                seg_profile.update(
                    width=seg_height,
                    height=seg_width,
                    transform=seg_transform,
                    compress="lzw",
                )

                segment_output_file = output_dir / f"{row}_{col}" / f"{band_name}.tif"
                segment_output_file.parent.mkdir(parents=True, exist_ok=True)

                with rasterio.open(
                    segment_output_file, "w", **seg_profile
                ) as segment_dst:
                    segment_dst.write(seg_data)

                mask_path = output_dir / f"{row}_{col}" / "mask.tif"
                if not mask_path.exists() and window_size == 768:
                    create_masks(aoi_gdf, mask_path, polygon, seg_profile)

                num_processed += 1

        print(
            f"Processed tiles: {num_processed}, blank tiles: {blank}, misshapen tiles: {misshapen}, no_mask: {no_mask}"
        )

        return num_processed, skipped, all_bounds
