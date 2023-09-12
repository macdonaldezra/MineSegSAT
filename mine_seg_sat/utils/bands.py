from pathlib import Path

from mine_seg_sat.constants import (
    MAX_RESOLUTION,
    MAX_RESOLUTION_SIZE,
    MID_RESOLUTION,
    MID_RESOLUTION_SIZE,
    MIN_RESOLUTION,
    MIN_RESOLUTION_SIZE,
)


def get_band_specification(filepath: Path) -> tuple[str, int]:
    """
    Get the metadata from a Sentinel-2 file.
    """
    for band in MIN_RESOLUTION:
        if band == filepath.stem:
            return (band, MIN_RESOLUTION_SIZE)
    for band in MID_RESOLUTION:
        if band == filepath.stem:
            return (band, MID_RESOLUTION_SIZE)
    for band in MAX_RESOLUTION:
        if band == filepath.stem:
            return (band, MAX_RESOLUTION_SIZE)

    return ("", 0)
