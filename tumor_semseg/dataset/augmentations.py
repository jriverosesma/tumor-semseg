import albumentations as A
import cv2


def get_training_augmentations(image_size: tuple[int, int]):
    return A.Compose(
        [
            # Crop or pad images to the right size
            A.OneOf(
                [
                    A.RandomSizedCrop(
                        min_max_height=(200, 256),
                        height=image_size[0],
                        width=image_size[1],
                        interpolation=cv2.INTER_LINEAR,
                        p=0.5,
                    ),
                    A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], p=0.5),
                ],
                p=1,
            ),
            # Basic flips and rotations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            # Geometric and perspective transformations
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            # Color adjustments
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.FancyPCA(alpha=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            # Noise and blur
            A.ISONoise(intensity=(0.1, 0.5), color_shift=(0.01, 0.05), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            # Lighting conditions
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.2
            ),
        ]
    )
