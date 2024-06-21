from .collator import FastTextCollator, ClassificationType, ClassificationCollator
from .classification_dataset import ClassificationDataset

AVAILABLE_COLLATORS = {
    "FastTextCollator": FastTextCollator,
    "ClassificationCollator": ClassificationCollator
}

AVAILABLE_DATASETS = {
    "ClassificationDataset": ClassificationDataset
}