import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            ToTensorV2(),
        ])