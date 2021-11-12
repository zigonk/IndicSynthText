import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from datawriter import FolderWriter
from synthgen import RendererV3
import random

# Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
# SECS_PER_IMG = 5 #max time per image in seconds
SECS_PER_IMG = None  # max time per image in seconds
# INSTANCE_PER_IMAGE = 900  # no. of times to use the same image
INSTANCE_PER_IMAGE = 10  # no. of times to use the same image
# path to the data-file, containing image, depth and segmentation:
SEED = 2001
DATA_PATH = Path('./aic_data/')
BACKGROUND_DIR = DATA_PATH / 'background'
FONT_DIR = DATA_PATH / 'fonts'
TEXT_PATH = DATA_PATH / 'vin-vnm.txt'
COLOR_MODEL_PATH = DATA_PATH / 'models' / 'colors_new.cp'
FONT_MODEL_PATH = DATA_PATH / 'models' / 'font_px2pt.pkl'


def main(lang, out_path, total_samples, viz=False):
    writer = FolderWriter(out_path, total_samples, word_box=True)
    writer.open()

    random.seed(SEED)
    np.random.seed(SEED)

    RV3 = RendererV3(COLOR_MODEL_PATH, FONT_DIR, TEXT_PATH, FONT_MODEL_PATH, max_time=SECS_PER_IMG)
    for i, info_path in enumerate(Path('outputs').glob('*.pkl')):
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
        img = io.imread(info['image_path'])
        depth = info['depth']
        seg = info['seg']
        area = info['area']
        label = info['label']

        try:
            res = RV3.render_text(img, depth, seg, area, label,
                                ninstance=INSTANCE_PER_IMAGE, viz=viz)
        except:
            continue

        # print(res)
        if len(res) > 0:
            writer.write(res)

        # visualize the output:
        if viz:
            plt.show(block=True)
            if 'q' == input('Continue? (q to quit)'):
                break

    writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz',
                        default=False, help='flag for turning on visualizations')
    parser.add_argument('--lang', dest='lang', default='en',
                        help='Generate synthetic scene-text images for language <lang>')
    parser.add_argument('--output_path', default='./',
                        help='path to store generated results')
    parser.add_argument('--total_samples', default=10000,
                        help='Total number of samples to generate')
    args = parser.parse_args()
    main(args.lang, args.output_path, args.total_samples, args.viz)

    cv2.destroyAllWindows()
