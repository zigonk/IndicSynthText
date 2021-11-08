# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

from pathlib import Path
import pygame
from pygame import freetype
from text_utils import FontState
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse


def gen_fontmodel(font_dir: Path, output_dir: Path):
    pygame.init()

    ys = np.arange(8, 200)
    A = np.c_[ys, np.ones_like(ys)]

    xs = []
    models = {}  # linear model

    FS = FontState(data_dir=Path('SynthTextGen'), create_model=True)
    print(FS)
    # plt.figure()
    # plt.hold(True)
    for i in range(len(FS.fonts)):
        print(i)
        font = freetype.Font(FS.fonts[i], size=12)
        h = []
        for y in ys:
            y = float(y)
            h.append(font.get_sized_glyph_height(y))
        h = np.array(h)
        m, _, _, _ = np.linalg.lstsq(A, h)
        print(m)
        print(font.name)
        models[font.name] = m
        xs.append(h)

    # with open(output_path+'/font_px2pt'+lang+'.cp', 'wb') as f:
    #     pickle.dump(models, f)
    # plt.plot(xs,ys[i])
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--font_dir', default='SynthTextGen/fonts')
    parser.add_argument('--output_dir', default='SynthTextGen/models')
    args = parser.parse_args()
    font_dir = Path(args.font_dir)
    output_dir = Path(args.output_dir)
    gen_fontmodel(font_dir, output_dir)
