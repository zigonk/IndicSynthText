# Author: Ankush Gupta, Loi Ly
# Date: 2015, 2021

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFont


def gen_fontmodel(font_dir: Path, output_dir: Path, show: bool = False):
    chars_dict = set(
        Path('SynthTextGen/newsgroup/newsgroupvn.txt').read_text('utf-8'))
    chars = ''.join(chars_dict)

    font_sizes = np.arange(8, 200)
    A = np.column_stack((font_sizes, np.ones_like(font_sizes)))

    models = {}  # linear model

    font_paths = sorted(font_dir.glob('**/*'))

    for font_path in font_paths:
        font_name = font_path.stem
        print(font_name, font_path)
        font_heights = []
        for font_size in font_sizes:
            font = ImageFont.truetype(str(font_path), size=font_size)
            font_heights.append(font.getsize(chars)[1])
        font_heights = np.array(font_heights)
        m, _, _, _ = np.linalg.lstsq(A, font_heights, rcond=None)
        print('>', m)
        models[font_name] = m

        if show:
            plt.title(font_name)
            plt.plot(font_sizes, font_heights)
            plt.xlabel('font_size (in points)')
            plt.ylabel('font_height (in pixels)')
            plt.show()

    # append language if other than 'en'
    output_path = output_dir / 'font_px2pt.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(models, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--font_dir', default='SynthTextGen/fonts')
    parser.add_argument('--output_dir', default='SynthTextGen/models')
    parser.add_argument('--lang', type=str, default='vn',
                        help='Language code. Ref: https://www.w3.org/International/articles/language-tags/')
    parser.add_argument('--show', action='store_true', default=False)
    args = parser.parse_args()
    font_dir = Path(args.font_dir)
    output_dir = Path(args.output_dir)
    gen_fontmodel(font_dir, output_dir, args.show)
