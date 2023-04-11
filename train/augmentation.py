import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transform():
    return A.Compose(
        [
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CenterCrop(p=1, height=224, width=224),
            A.OneOf([
                    A.ElasticTransform(border_mode=1),
                    A.GridDistortion(p=1),
                    A.GaussNoise(p=1),
                    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=1.0),
            A.OneOf([
                    A.ElasticTransform(border_mode=1),
                    A.GridDistortion(p=1),
                    A.GaussNoise(p=1),
                    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

def val_transform():
    return A.Compose(
        [
            A.Resize(height=256, width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
