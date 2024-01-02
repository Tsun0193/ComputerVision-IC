import os

LOC_SYNSET_MAPPING = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "imagenet",
    "LOC_synset_mapping.txt",
)

LOC_VAL_SOLUTION = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "imagenet",
    "LOC_val_solution.csv",
)

BATCH_SIZE = 256

BASE_URL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "imagenet",
    "imagenet-val",
)

FIG_SIZE = 10

NORMALIZE_MEAN = (0.5, 0.5, 0.5)  # Mean value for R, G, B channel of images

NORMALIZE_STD = (0.5, 0.5, 0.5)  # Standard deviation for R, G, B channel of images

ALEXNET_IMG_SIZE = 227

IMG_SIZE = 224

N_IMAGES = 25

OUTPUT_DIM = 6
