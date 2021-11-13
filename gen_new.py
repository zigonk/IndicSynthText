import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from datawriter import FolderWriter, ICDAR2015Writer
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

def main(data_dir, info_dir, font_dir, text_path, out_path, total_samples, viz=False):
    writer = ICDAR2015Writer(out_path, total_samples)
    writer.open()

    random.seed(SEED)
    np.random.seed(SEED)

    DATA_PATH = Path(data_dir)
    FONT_DIR = Path(font_dir)
    TEXT_PATH = Path(text_path)
    COLOR_MODEL_PATH = DATA_PATH / 'models' / 'colors_new.cp'
    FONT_MODEL_PATH = DATA_PATH / 'models' / 'font_px2pt.pkl'

    RV3 = RendererV3(COLOR_MODEL_PATH, FONT_DIR, TEXT_PATH, FONT_MODEL_PATH, max_time=SECS_PER_IMG)
    for i, info_path in enumerate(Path(info_dir).glob('*.pkl')):
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
        img = info['image']
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
    parser.add_argument('data_dir')
    parser.add_argument('info_dir')
    parser.add_argument('font_dir')
    parser.add_argument('text_path')
    parser.add_argument('--viz', action='store_true', dest='viz',
                        default=False, help='flag for turning on visualizations')
    parser.add_argument('--output_path', default='./',
                        help='path to store generated results')
    parser.add_argument('--total_samples', default=10000,
                        help='Total number of samples to generate')
    args = parser.parse_args()
    main(args.data_dir, args.info_dir, args.font_dir, args.text_path, args.output_path, args.total_samples, args.viz)

    cv2.destroyAllWindows()
