import os

# Base Paths
# Base Paths
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "B. Disease Grading")
IMAGES_DIR = os.path.join(DATASET_DIR, "1. Original Images")
GROUNDTRUTHS_DIR = os.path.join(DATASET_DIR, "2. Groundtruths")

TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, "a. Training Set")
TEST_IMAGES_DIR = os.path.join(IMAGES_DIR, "b. Testing Set")

TRAIN_LABELS_PATH = os.path.join(GROUNDTRUTHS_DIR, "a. IDRiD_Disease Grading_Training Labels.csv")
TEST_LABELS_PATH = os.path.join(GROUNDTRUTHS_DIR, "b. IDRiD_Disease Grading_Testing Labels.csv")

# Segmentation Paths
SEG_TRAIN_DIR = os.path.join(DATASET_DIR.replace("B. Disease Grading", "A. Segmentation"), "1. Original Images", "a. Training Set")
SEG_TEST_DIR = os.path.join(DATASET_DIR.replace("B. Disease Grading", "A. Segmentation"), "1. Original Images", "b. Testing Set")
SEG_MASKS_TRAIN_DIR = os.path.join(DATASET_DIR.replace("B. Disease Grading", "A. Segmentation"), "2. All Segmentation Groundtruths", "a. Training Set")
SEG_MASKS_TEST_DIR = os.path.join(DATASET_DIR.replace("B. Disease Grading", "A. Segmentation"), "2. All Segmentation Groundtruths", "b. Testing Set")

# Localization Paths
LOC_TRAIN_DIR = SEG_TRAIN_DIR # Uses same images usually, checking... Actually Localization has its own images folder in C. Localization/1. Original Images
LOC_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "C. Localization", "1. Original Images")
LOC_TRAIN_IMAGES = os.path.join(LOC_IMAGES_DIR, "a. Training Set")
LOC_TEST_IMAGES = os.path.join(LOC_IMAGES_DIR, "b. Testing Set")

LOC_GT_DIR = os.path.join(BASE_DIR, "dataset", "C. Localization", "2. Groundtruths")
LOC_OD_TRAIN_CSV = os.path.join(LOC_GT_DIR, "1. Optic Disc Center Location", "a. IDRiD_OD_Center_Training Set_Markups.csv")
LOC_OD_TEST_CSV = os.path.join(LOC_GT_DIR, "1. Optic Disc Center Location", "b. IDRiD_OD_Center_Testing Set_Markups.csv")
LOC_FOVEA_TRAIN_CSV = os.path.join(LOC_GT_DIR, "2. Fovea Center Location", "IDRiD_Fovea_Center_Training Set_Markups.csv")
LOC_FOVEA_TEST_CSV = os.path.join(LOC_GT_DIR, "2. Fovea Center Location", "IDRiD_Fovea_Center_Testing Set_Markups.csv")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "efficientnet_b0_dr.pth")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100 # Extended training
NUM_CLASSES = 5
IMAGE_SIZE = 300
RANDOM_SEED = 42
