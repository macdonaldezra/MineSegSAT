import random
import typing as T
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
import torch

from mine_seg_sat.constants import (MAX_RESOLUTION, MID_RESOLUTION,
                                    MIN_RESOLUTION)


class MineSATDataset(torch.utils.data.Dataset):

    """
    Dataset for collection of binary masks and Sentinel-2 tiles containing mines.
    """

    def __init__(
        self,
        split: str,
        data_path: Path,
        transformations: T.Any = None,
        preprocessing: T.Any = None,
        flatten_mask: bool = False,
        use_mask: bool = True,
        rescale: bool = True,
        min_max_normalization: bool = True,
        max_values: T.Optional[T.List[float]] = None,
    ):
        """
        Parameters:
            split: Split to use, one of 'train', 'val', 'test'.
            data_path: The root directory that contains the filepaths to the images and masks.
            transformations: Transformations to apply to the images and masks.
            flatten_mask: Whether to flatten the mask into a single channel.
            use_mask: Whether to return a mask along with the image.
            rescale: Whether or not to rescale all lower resolution bands up to the 10m resolution.
            max_values: Maximum values for each band to use for rescaling. If None, then norm
        """
        self.data_path = data_path
        self.split = split
        assert self.split in [
            "train",
            "val",
            "test",
        ], f"Split must be one of 'train', 'val', 'test', got {self.split}"
        assert (
            self.data_path / "dataset_splits.csv"
        ).exists(), "dataset_splits.csv not found in data_path."
        self.use_mask = use_mask
        self.rescale = rescale
        self.min_max_normalization = min_max_normalization
        if max_values is not None:
            self.max_values = np.array(max_values)
            self.min_values = np.zeros_like(self.max_values)

        self.df = pd.read_csv(self.data_path / "dataset_splits.csv")
        self.filepaths = self.df.loc[
            self.df["split"] == self.split, "data_path"
        ].tolist()
        self.maskpaths = self.df.loc[
            self.df["split"] == self.split, "mask_path"
        ].tolist()
        assert len(
            self.filepaths) > 0, f"No files found for split {self.split}"
        self.data_path = data_path
        self.index_to_rgb = {
            0: (0, 0, 0),
            1: (212, 0, 0),
        }

        self.transformations = transformations
        self.preprocessing = preprocessing
        self.flatten_mask = flatten_mask
        self.num_classes = 1

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index: int, return_original: bool = False):
        """
        Return the 12 bands of the image and the mask. Note that if the image is not 10m resolution,
        then the lower resolution bands will be rescaled to 10m resolution if self.rescale is True.
        Otherwise, the lower resolution bands will be returned as is.
        """
        filepath = self.filepaths[index]
        maskpath = self.maskpaths[index]
        high_resolution_bands = np.stack(
            [
                tiff.imread(
                    f"{self.data_path.as_posix()}/{filepath}/{band}.tif")
                for band in MAX_RESOLUTION
            ]
        )
        mid_resolution_bands = np.stack(
            [
                tiff.imread(
                    f"{self.data_path.as_posix()}/{filepath}/{band}.tif")
                for band in MID_RESOLUTION
            ]
        )
        min_resolution_bands = np.stack(
            [
                tiff.imread(
                    f"{self.data_path.as_posix()}/{filepath}/{band}.tif")
                for band in MIN_RESOLUTION
            ]
        )
        mask = tiff.imread(f"{self.data_path.as_posix()}/{maskpath}/mask.tif").astype(
            "int16"
        )

        if self.rescale:
            # Rescale lower resolution bands to 10m resolution
            mid_resolution_bands = np.stack(
                [
                    cv2.resize(band, None, fx=2, fy=2,
                               interpolation=cv2.INTER_CUBIC)
                    for band in mid_resolution_bands
                ]
            )
            min_resolution_bands = np.stack(
                [
                    cv2.resize(band, None, fx=6, fy=6,
                               interpolation=cv2.INTER_CUBIC)
                    for band in min_resolution_bands
                ]
            )
            all_bands = np.concatenate(
                [high_resolution_bands, mid_resolution_bands, min_resolution_bands],
                axis=0,
            ).astype("int16")
        else:
            all_bands = [
                high_resolution_bands,
                mid_resolution_bands,
                min_resolution_bands,
            ]

        if callable(self.transformations) and not return_original:
            if isinstance(all_bands, np.ndarray):
                all_bands = all_bands.transpose(
                    1, 2, 0)  # (C, H, W) -> (H, W, C)
                if self.min_max_normalization:
                    all_bands = self.min_max_normalize(all_bands)
                data = self.transformations(image=all_bands, mask=mask)
                all_bands = data["image"].to(dtype=torch.float32)
                # all_bands = data["image"].transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
                mask = data["mask"].to(dtype=torch.long)
            else:
                all_bands[0], all_bands[1], all_bands[2], mask = self.transformations(
                    high=all_bands[0], mid=all_bands[1], low=all_bands[2], mask=mask
                )

        if self.flatten_mask:
            mask = torch.nn.functional.one_hot(
                mask.to(dtype=torch.long), self.num_classes
            ).permute(2, 0, 1)

        return (all_bands, mask)

    def min_max_normalize(self, image: np.ndarray) -> np.ndarray:
        # Get the minimum and maximum values for each band
        return (image - self.min_values) / (self.max_values - self.min_values)

    def colorize_mask(self, mask: np.ndarray):
        """
        Given a mask, colorize it according to the legend.
        """
        rgb_mask = np.zeros(mask.shape[:2] + (3,))
        rgb_mask[mask == 0] = (0, 0, 0)
        for index, color in self.index_to_rgb.items():
            rgb_mask[mask == index] = color

        return np.uint8(rgb_mask)

    def get_image_and_mask(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the image at the given index.
        """
        filepath = self.filepaths[index]
        maskpath = self.maskpaths[index]
        high_resolution_bands = np.clip(
            self.scale_band(
                np.stack(
                    [
                        tiff.imread(
                            f"{self.data_path.as_posix()}/{filepath}/{band}.tif"
                        )
                        for band in ["B04", "B03", "B02"]
                    ],
                    axis=-1,
                )
            ),
            0,
            1,
        )
        mask = tiff.imread(f"{self.data_path.as_posix()}/{maskpath}/mask.tif")

        return (high_resolution_bands, mask)

    def show_images_with_class(self, num_images: int = 5):
        """
        Show images for the class that ends up appearing in the dataset.
        """
        filepaths = self.df[["data_path", "mask_path"]].values.tolist()
        selected = random.sample(filepaths, num_images)
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 4))

        for i, paths in enumerate(selected):
            image = np.stack(
                [
                    np.clip(
                        self.scale_band(
                            tiff.imread(
                                f"{self.data_path.as_posix()}/{paths[0]}/{band}.tif"
                            )
                        ),
                        0,
                        1,
                    )
                    for band in ["B04", "B03", "B02"]
                ],
                axis=-1,
            )
            mask = tiff.imread(
                f"{self.data_path.as_posix()}/{paths[1]}/mask.tif")
            title = self.parse_title(i)

            axes[i, 0].imshow(image)
            axes[i, 0].set_title(title)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask)
            axes[i, 1].set_title("Mask")
            axes[i, 1].axis("off")

        fig.tight_layout()
        plt.show()

    def scale_band(self, band: np.ndarray, percentile: float = 95) -> np.ndarray:
        """
        Scale the band to 0-1.
        """
        return band / np.percentile(band, percentile)

    def display_image_and_mask(self, index: int):
        """
        Given an image index, display the image and mask.
        """
        rgb, mask = self.get_image_and_mask(index)
        mask = self.colorize_mask(mask)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.tight_layout()

        # Display the image
        axes[0].imshow(rgb)
        title = self.parse_title(index)
        axes[0].set_title(title)
        axes[0].axis("off")

        # Display the mask
        axes[1].imshow(mask)
        axes[1].set_title("Mask")
        axes[1].axis("off")

        plt.show()

    def parse_title(self, index: int, show_tile: bool = False) -> str:
        path = self.filepaths[index]
        title = ""
        if "ab_mines" in path:
            title += "Alberta"
        elif "bc_mines" in path:
            title += "British Columbia"

        if show_tile:
            title += " " + path.split("/")[-1]
        if "resource" in self.df.columns:
            title += (
                " "
                + self.df.loc[self.df.data_path == self.filepaths[index]][
                    "resource"
                ].tolist()[0]
            )
            titles = title.split(";")
            if len(titles) == 2:
                title = " & ".join(titles)
            elif len(titles) > 2:
                title = ", ".join(titles[:-1]) + " & " + titles[-1]

        return title

    def display_model_output(self, index: int, predicted_mask: np.ndarray):
        """
        Given an image index, display the image, mask, and predicted mask.
        """
        image, mask = self.get_image_and_mask(index)
        if self.transformations is not None:
            data = self.transformations(image=image, mask=mask.astype("int16"))
            image = data["image"].numpy().transpose(1, 2, 0)
            mask = data["mask"].numpy()

        mask = self.colorize_mask(mask)
        predicted_mask = self.colorize_mask(predicted_mask)

        # Create a color image using RGB bands
        fig, axes = plt.subplots(1, 3, figsize=(8, 4))

        # Display the image
        axes[0].imshow(image)
        title = self.parse_title(index)
        axes[0].set_title(title)
        axes[0].axis("off")

        # Display the mask
        axes[1].imshow(mask)
        axes[1].set_title("Mask")
        axes[1].axis("off")

        # Display the predicted mask
        axes[2].imshow(predicted_mask)
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        fig.tight_layout()
        plt.show()

    # @staticmethod
    # def display_images(image_dict):
    #     fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    #     fig.tight_layout()

    #     for i, (title, image) in enumerate(image_dict.items()):
    #         row = i // 3
    #         col = i % 3
    #         axes[row, col].imshow(image)
    #         axes[row, col].set_title(title)
    #         axes[row, col].axis("off")

    #     plt.show()
    @staticmethod
    # def display_images(image_dict):
    #     num_images = len(image_dict)
    #     # Calculate the number of rows based on the number of images
    #     num_rows = int(np.ceil(num_images / 3.0))
    #     fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    #     fig.tight_layout()
    #     for i, (title, image) in enumerate(image_dict.items()):
    #         row = i // 3
    #         col = i % 3
    #         if num_rows == 1:  # If there's only one row, axes is a 1D array
    #             ax = axes[col]
    #         else:
    #             ax = axes[row, col]
    #         ax.imshow(image)
    #         ax.set_title(title)
    #         ax.axis("off")
    #     plt.show()
    def display_images(image_dict):
        num_images = len(image_dict)
        # Calculate the number of rows based on the number of images
        num_rows = int(np.ceil(num_images / 3.0))
        fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
        fig.tight_layout()

        for i, (title, image) in enumerate(image_dict.items()):
            row = i // 3
            col = i % 3
            if num_rows == 1:  # If there's only one row, axes is a 1D array
                ax = axes[col]
            else:
                ax = axes[row, col]

            # Adjust colormap and normalization for NBR
            if title == "NBR":
                im = ax.imshow(image, cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax, orientation='vertical')
            else:
                ax.imshow(image)

            ax.set_title(title)
            ax.axis("off")

        plt.show()

    def display_transformed_images(self, index: int, percentile: int = 95):
        """
        Generate Sentinel-2 images from the given filepath. The returned images are as follows:

        NDVI: Normalized Difference Vegetation Index
        NDBI: Normalized Difference Built-up Index
        NDWI: Normalized Difference Water Index
        False Color: B08, B04, B03
        Mask: The class labels for each pixel
        """
        # Unpack the bands
        filepath = self.filepaths[index]
        maskpath = self.maskpaths[index]
        B02 = self.scale_band(
            tiff.imread(f"{self.data_path.as_posix()}/{filepath}/B02.tif"),
            percentile=percentile,
        )
        B03 = self.scale_band(
            tiff.imread(f"{self.data_path.as_posix()}/{filepath}/B03.tif"),
            percentile=percentile,
        )
        B04 = self.scale_band(
            tiff.imread(f"{self.data_path.as_posix()}/{filepath}/B04.tif"),
            percentile=percentile,
        )
        B07 = self.scale_band(
            tiff.imread(f"{self.data_path.as_posix()}/{filepath}/B07.tif"),
            percentile=percentile,
        )
        B08 = self.scale_band(
            tiff.imread(f"{self.data_path.as_posix()}/{filepath}/B08.tif"),
            percentile=percentile,
        )

        B11 = cv2.resize(
            tiff.imread(f"{self.data_path.as_posix()}/{filepath}/B11.tif"),
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )
        B12 = cv2.resize(
            tiff.imread(f"{self.data_path.as_posix()}/{filepath}/B12.tif"),
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )
        B11 = self.scale_band(B11, percentile=percentile)
        B12 = self.scale_band(B12, percentile=percentile)

        mask = tiff.imread(f"{self.data_path.as_posix()}/{maskpath}/mask.tif")
        mask = self.colorize_mask(mask)

        # Calculate NDVI (Normalized Difference Vegetation Index)
        NDVI = (B08 - B04) / (B08 + B04)

        # Calculate NDBI (Normalized Difference Built-Up Index)
        NDBI = (B11 - B08) / (B11 + B08)

        # Calculate NDWI (Normalized Difference Water Index)
        NDWI = (B08 - B12) / (B08 + B12)

        # Create a color image using RGB bands
        RGB = np.stack([B04, B03, B02], axis=-1)

        # Upscale B07 to match the resolution of B04
        B07_rescaled = cv2.resize(
            B07, (B04.shape[1], B04.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Now compute NBR
        NBR = (B08 - B07_rescaled) / (B08 + B07_rescaled)
        # Create a false-color composite image
        false_color = np.stack([B08, B04, B03], axis=-1)

        self.display_images(
            {
                "NBR": NBR,
                "NDVI": NDVI,
                "NDBI": NDBI,
                "NDWI": NDWI,
                "RGB": RGB,
                "False Color": false_color,
                "Mask": mask,
            }
        )


    def downsample(self, image, scale_factor):

        if not (0 < scale_factor < 1):
            raise ValueError("Scale factor must be between 0 and 1.")
        dimensions = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        downsampled_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
        return downsampled_image

    def upsample(self, image, target_shape):
        upsampled_image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
        return upsampled_image

    def get_numerical_values(self, index: int, percentile: int = 95, scale_factor: float = 0.05):
    # Assuming that all the necessary functions and variables are defined elsewhere in the class
        filepath = self.filepaths[index]

        # Load and scale the bands
        B04 = self.downsample(self.scale_band(tiff.imread(f"{self.data_path.as_posix()}/{filepath}/B04.tif"), percentile=percentile), scale_factor)
        B07 = self.downsample(self.scale_band(tiff.imread(f"{self.data_path.as_posix()}/{filepath}/B07.tif"), percentile=percentile), scale_factor)
        B08 = self.downsample(self.scale_band(tiff.imread(f"{self.data_path.as_posix()}/{filepath}/B08.tif"), percentile=percentile), scale_factor)

        # Calculate NDVI and NBR indices
        NDVI = (B08 - B04) / (B08 + B04)
        B07_upsampled = self.upsample(B07, B08.shape)
        NBR = (B08 - B07_upsampled) / (B08 + B07_upsampled)

        # Return as dictionary
        return {
            "NDVI": NDVI,
            "NBR": NBR
        }
