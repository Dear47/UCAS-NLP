import os

ROOT_PATH = os.getcwd()
DATA_PATH = os.path.join(ROOT_PATH,"data")
SOURCE_FILE = os.path.join(DATA_PATH,"news.2017.zh.shuffled.deduped")
TRAIN_FILE = os.path.join(DATA_PATH, "train.txt")
VALID_FILE = os.path.join(DATA_PATH, "valid.txt")
CHECKPOINT_PATH = os.path.join(ROOT_PATH, "checkpoints")
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
VOCAB_FILE = os.path.join(DATA_PATH, "vocab.json")
