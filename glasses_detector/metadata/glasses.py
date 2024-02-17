from pathlib import Path

import glasses_detector.metadata.shared as shared

RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "glasses"
RAW_DATA_FILENAME = RAW_DATA_DIRNAME / "glasses.zip"
RAW_DATA_IMAGE_DIRNAME = RAW_DATA_DIRNAME / "MeGlass_120x120"
RAW_DATA_LABELS_FILENAME = RAW_DATA_DIRNAME / "labels.txt"
PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "glasses"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'glasses_processed.h5'
ESSENTIALS_FILENAME = shared.DATA_DIRNAME / "processed" / "glasses_essentials.json"

#IMAGES_URL = 'https://drive.usercontent.google.com/download?id=1V0c8p6MOlSFY5R-Hu9LxYZYLXd8B8j9q&export=download&authuser=0&confirm=t&uuid=ac6c4636-cc3f-4290-aee4-21fd0f324ce9&at=APZUnTXovugZps4XBv3Qhxak4Gg_%3A1707861702490'
IMAGES_URL = 'http://tinyurl.com/ycykr2eh'
LABELS_URL = 'https://raw.githubusercontent.com/cleardusk/MeGlass/master/meta.txt'

INPUT_DIMS = (1, 120, 120)
OUTPUT_DIMS = (1,)
MAPPING = {0: 'No Glasses', 1: 'Glasses'}
