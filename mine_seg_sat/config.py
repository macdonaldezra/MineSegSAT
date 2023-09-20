import json
import os
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field
from ruamel.yaml import YAML
from ruamel.yaml.composer import ComposerError
from ruamel.yaml.parser import ParserError
from ruamel.yaml.scanner import ScannerError

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "output"


class PreProcessingConfig(BaseModel):
    data_directory: Path = Field(DATA_PATH, env="DATA_PATH")
    output_directory: Path = Field(OUTPUT_PATH, env="OUTPUT_PATH")
    aoi_directory: Path = Field(DATA_PATH / "aoi", env="AOI_PATH")
    preprocessor_executable_path: Path = Field(..., env="PREPROCESSOR_EXECUTABLE_PATH")
    clc_gdb_path: Path = Field(
        DATA_PATH / "u2018_clc2018_v2020_20u1_fgdb",
        env="CLC_PATH",
    )
    base_resolution: int = 128  # 128 x 128 for 60m resolution
    download_timeout: int = 60 * 60  # 1 hour
    sentinel_username: str = Field(..., env="SENTINEL_USERNAME")
    sentinel_password: str = Field(..., env="SENTINEL_PASSWORD")
    sentinel_trigger_download_limit: int = (
        20  # Number of files allowed to trigger the download of at a time.
    )
    clc_legend_path: Path = Field(..., env="CLC_LEGEND_PATH")


class TrainingConfig(BaseModel):
    image_size: int
    num_classes: int
    epochs: int
    lr_scheduler: str
    learning_rate: float
    max_learning_rate: Optional[float] = 0.05
    min_learning_rate: Optional[float] = 1e-5
    learning_rate_weight_decay: float
    batch_size: int
    save_frequency: int
    class_weights: Optional[List[float]] = None
    weight_filepath: Union[None, Path] = None
    ignore_index: Optional[int] = None
    model: str
    in_channels: int = 12
    encoder: Optional[str] = "resnet50"
    num_dataloader_workers: int
    profile: bool = False
    has_aux_classifier: bool = False
    loss: str
    dataset: str = "eusegsat"
    comment: Optional[str] = None
    included_classes: Optional[List[int]] = None

    data_path: Path = Field(DATA_PATH, env="DATA_PATH")
    output_path: Path = Field(..., env="OUTPUT_PATH")

    # AWS credentials
    aws_access_key_id: str = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(None, env="AWS_SECRET_ACCESS_KEY")
    aws_s3_bucket: str = Field(None, env="AWS_S3_BUCKET")


def parse_yaml(data: Union[IO, bytes]) -> Union[None, Dict[str, Any]]:
    """
    Parse bytes or input data that ideally contains valid yaml.
    """
    try:
        yaml = YAML(typ="safe")
        return yaml.load(data)
    except (ScannerError, ParserError) as err:
        print(f"Error while trying to parse YAML:\n {err}")
        return None
    except ComposerError as err:
        print(f"Provided more than one YAML document:\n {err}")
        return None


def read_yaml(filepath: Path) -> Union[None, Dict[str, Any]]:
    """
    Read in a YAML file and return file contents in a dict.
    """
    try:
        fptr = open(filepath, "r")
        data = parse_yaml(fptr)
    except FileNotFoundError as err:
        print(f"File {err.filename} not found.")
        return None
    except IOError as err:
        print(f"Unable to parse contents of {err.filename}.")
        return None

    return data


def config_to_yaml(config: TrainingConfig, filepath: Path) -> None:
    """
    Output a config object to a yaml file.

    Args:
        filepath (Path): Path to the desired output file.
    """
    print(f"Outputting config object to {filepath.as_posix()}")
    filepath.parent.mkdir(exist_ok=True, parents=True)
    config_dict = json.loads(config.json())
    try:
        file = open(filepath, "w")
        yaml.dump(config_dict, file)
        file.close()
    except Exception as err:
        print(f"Unable to open {filepath} for writing.\n\n{err}")


def get_preprocessing_config() -> PreProcessingConfig:
    """
    Try to read in a YAML config file from the environment variable PREPROCESSING_CONFIG.

    If that file doesn't exist, then return the config file found in config_files/default.yaml.
    """
    if "PREPROCESSING_CONFIG" not in os.environ:
        print(
            "Variable PREPROCESSING_CONFIG not found. Falling back to default config file from project root."
        )

    config_path = Path(os.getenv("PREPROCESSING_CONFIG", PROJECT_ROOT / "default.yaml"))
    try:
        config_data = read_yaml(config_path)
    except OSError:
        print("Unable to parse config from provided filepath.")
        raise ValueError("Unable to load settings.")

    if not config_data:
        print(
            "Returned config is empty. Please check the format of your config file and try again."
        )

    config = PreProcessingConfig.parse_obj(config_data)
    assert (
        config.data_directory.exists()
    ), f"Data directory {config.data_directory} does not exist."
    assert (
        config.aoi_directory.exists()
    ), f"AOI directory {config.aoi_directory} does not exist."
    assert (
        config.aoi_directory.exists()
    ), f"AOI directory {config.aoi_directory} does not exist."
    assert (
        config.clc_gdb_path.exists()
    ), f"CLC gdb {config.clc_gdb_path} does not exist."
    assert (
        config.preprocessor_executable_path.exists()
    ), f"Preprocessor executable {config.preprocessor_executable_path} does not exist."

    return config


def get_model_config() -> TrainingConfig:
    """
    Try and parse a YAML file and return the YAML file parsed as a Training Config object.
    """
    if "MODEL_CONFIG" not in os.environ:
        print(
            "Variable MODEL_CONFIG not found. Falling back to default config file from project root."
        )

    config_path = Path(os.getenv("MODEL_CONFIG", PROJECT_ROOT / "model_default.yaml"))
    try:
        config_data = read_yaml(config_path)
    except OSError:
        print("Unable to parse config from provided filepath.")
        raise ValueError("Unable to load model settings.")

    if not config_data:
        print(
            "Returned config is empty. Please check the format of your config file and try again."
        )

    config = TrainingConfig.parse_obj(config_data)

    return config
