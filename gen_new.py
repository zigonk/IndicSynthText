from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform

from datawriter import FolderWriter, ICDAR2015Writer
from synthgen import RendererV3
import random

# Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
# SECS_PER_IMG = 5 #max time per image in seconds
SECS_PER_IMG = None  # max time per image in seconds
# INSTANCE_PER_IMAGE = 900  # no. of times to use the same image
INSTANCE_PER_IMAGE = 5  # no. of times to use the same image
# path to the data-file, containing image, depth and segmentation:
SEED = 2001

def main(bg_dir: Path, depth_dir: Path, seg_dir: Path, font_dir: Path,
         text_path: Path, output_dir: Path, total_samples, viz):
    writer = ICDAR2015Writer(output_dir, total_samples)
    writer.open()

    random.seed(SEED)
    np.random.seed(SEED)

    color_model_path = model_dir / 'colors_new.cp'
    font_model_path = model_dir / 'font_px2pt.pkl'

    RV3 = RendererV3(color_model_path, font_dir, text_path, font_model_path, max_time=SECS_PER_IMG)
    for i, image_path in enumerate(bg_dir.iterdir()):
        image_name = image_path.stem
        print('Processing', image_path)

        depth_path = depth_dir / (image_name + '.npz')
        if not depth_path.exists():
            print(depth_path, 'does not exist. Skip')
            continue

        seg_path = seg_dir / (image_name + '.npz')
        if not seg_path.exists():
            print(seg_path, 'does not exist. Skip')
            continue

        img = io.imread(str(image_path))
        with np.load(depth_path) as data:
            depth = data['depth']
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = 1 - depth
            depth = depth * 255
        with np.load(seg_path) as data:
            seg = data['seg']
            area = data['area']
            label = data['label']

        try:
            res = RV3.render_text(img, depth, seg, area, label,
                                    ninstance=INSTANCE_PER_IMAGE, viz=viz)
        except Exception as e:
            print(f'[ERROR] {image_path}: {e}')

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
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--bg_dir', type=Path, default=None)
    parser.add_argument('--depth_dir', type=Path, default=None)
    parser.add_argument('--seg_dir', type=Path, default=None)
    parser.add_argument('--font_dir', type=Path, default=None)
    parser.add_argument('--text_path', type=Path, default=None)
    parser.add_argument('--model_dir', type=Path, default=None)
    parser.add_argument('--viz', action='store_true', dest='viz',
                        default=False, help='flag for turning on visualizations')
    parser.add_argument('--output_dir', default='outputs', type=Path,
                        help='path to store generated results')
    parser.add_argument('--total_samples', default=10000,
                        help='Total number of samples to generate')
    args = parser.parse_args()

    bg_dir = args.bg_dir or Path(args.data_dir) / 'bg'
    depth_dir = args.depth_dir or Path(args.data_dir) / 'depths'
    seg_dir = args.seg_dir or Path(args.data_dir) / 'segs'
    font_dir = args.font_dir or Path(args.data_dir) / 'fonts'
    text_path = args.text_path or Path(args.data_dir) / 'text.txt'
    model_dir = args.model_dir or Path(args.data_dir) / 'models'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    main(bg_dir, depth_dir, seg_dir, font_dir, text_path, output_dir, args.total_samples, args.viz)

    cv2.destroyAllWindows()
