import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_transforms(dim: int) -> A.Compose:
    """
    Return a composition of Albumentations transforms for training data.
    """
    transforms = [
        A.RandomCrop(height=dim, width=dim, always_apply=True),
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ]

    return A.Compose(transforms, is_check_shapes=False)


def get_validation_transforms(dim: int) -> A.Compose:
    """
    Return a composition of Albumentations transforms for validation data.
    """
    return A.Compose(
        [A.RandomCrop(height=dim, width=dim, always_apply=True), ToTensorV2()],
        is_check_shapes=False,
    )


def get_test_transforms(dim: int) -> A.Compose:
    """
    Return a composition of Albumentations transforms for testing data.
    """
    return A.Compose(
        [A.CenterCrop(height=dim, width=dim, always_apply=True), ToTensorV2()],
        is_check_shapes=False,
    )
