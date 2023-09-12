# Spatial Resolution Data Found Here: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
# 10 meter resolution bands, order is important for Max resolution bands, B04=Red, B03=Green, B02=Blue, B08=NIR
MAX_RESOLUTION = ["B04", "B03", "B02", "B08"]
MID_RESOLUTION = [
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    # "scl",
]  # 20 meter resolution bands
MIN_RESOLUTION = ["B01", "B09"]  # 60 meter resolution bands
# MIN_RESOLUTION = ["B01", "B09", "aot"]  # 60 meter resolution bands

MIN_RESOLUTION_SIZE = 128
MID_RESOLUTION_SIZE = 384
MAX_RESOLUTION_SIZE = 768

EXTRACTED_BANDS = [
    ("B02", 10),
    ("B03", 10),
    ("B04", 10),
    ("B08", 10),
    ("B05", 20),
    ("B06", 20),
    ("B07", 20),
    ("B8A", 20),
    ("B11", 20),
    ("B12", 20),
    # ("scl", 20),
    ("B01", 60),
    ("B09", 60),
    # ("aot", 60),
]


# This ordering is based on the order of the bands read into the dataloader
# Basically - read 10m, 20m, 60m bands in that order
BAND_ORDERING = [
    "B04",
    "B03",
    "B02",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]
