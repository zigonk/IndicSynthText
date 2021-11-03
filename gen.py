# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""
from datawriter import LMDBDataWriter, FolderWriter
import io
from typing import Dict, List
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import h5py
import lmdb
import os
import sys
import traceback
import os.path as osp
from synthgen import *
from common import *
import tarfile


# Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
# SECS_PER_IMG = 5 #max time per image in seconds
SECS_PER_IMG = None  # max time per image in seconds
# INSTANCE_PER_IMAGE = 900  # no. of times to use the same image
INSTANCE_PER_IMAGE = 10  # no. of times to use the same image
# path to the data-file, containing image, depth and segmentation:
DATA_PATH = './SynthTextGen/'
DB_FNAME = osp.join(DATA_PATH, 'dset.h5')

# @azhar


def filter_text(lang, text):
    unicode_range = {'odia': '[^\u0020-\u0040-\u0B00-\u0B7F]', 'kanada': '[^\u0020-\u0040-\u0C80-\u0CFF]',
                     'tamil': '[^\u0020-\u0040-\u0B80-\u0BFF]', 'malyalam': '[^\u0020-\u0040-\u0D00-\u0D7F]',
                     'urdu': '[^\u0020-\u0040-\u0600-\u06FF]', 'telgu': '[^\u0020-\u0040-\u0C00-\u0C7F]',
                     'marathi': '[^\u0020-\u0040-\u0900-\u097F]', 'sanskrit': '[^\u0020-\u0040-\u0900-\u097F]',
                     'hindi': '[^\u0020-\u0040-\u0900-\u097F]', 'ban': '[^\u0020-\u0040-\u0980-\u09FF]'}
    import re
    t = re.sub(unicode_range[lang], '', text)
    if len(text) == len(t):
        return False
    else:
        return True


def main(lang, out_path, total_samples, viz=False):
    if osp.exists(DB_FNAME):
        db = h5py.File(DB_FNAME, 'r')
    else:
        print(colorize(
            Color.RED, f'Data not found at {DB_FNAME}. Download from https://www.kaggle.com/azharshaikh/synthtextgen', bold=True))
        sys.stdout.flush()
        sys.exit(-1)

    print(colorize(Color.BLUE, '\t-> done', bold=True))

    # get the names of the image files in the dataset:
    imnames = sorted(db['image'].keys())
    N = len(imnames)
    global NUM_IMG
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx, end_idx = 0, min(NUM_IMG, N)

    # writer = LMDBDataWriter(out_path, total_samples)
    writer = FolderWriter(out_path, total_samples, word_box=True)
    writer.open()

    RV3 = RendererV3(DATA_PATH, lang, max_time=SECS_PER_IMG)
    for i in range(start_idx, end_idx):
        imname = imnames[i]
        try:
            # get the image:
            img = Image.fromarray(db['image'][imname][:])
            # get the pre-computed depth:
            #  there are 2 estimates of depth (represented as 2 "channels")
            #  here we are using the second one (in some cases it might be
            #  useful to use the other one):
            depth = db['depth'][imname][:].T
            depth = depth[:, :, 1]
            # get segmentation:
            seg = db['seg'][imname][:].astype('float32')
            area = db['seg'][imname].attrs['area']
            label = db['seg'][imname].attrs['label']
            # print(label)

            # re-size uniformly:
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz, Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))

            print(colorize(Color.RED, '%d of %d' % (i, end_idx-1), bold=True))
            res = RV3.render_text(img, depth, seg, area, label,
                                  ninstance=INSTANCE_PER_IMAGE, viz=viz)
            # print(res)
            if len(res) > 0:
                writer.write(res)

            # visualize the output:
            if viz:
                if 'q' in input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True)):
                    break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            break

    writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz',
                        default=False, help='flag for turning on visualizations')
    parser.add_argument('--lang', dest='lang', required=True,
                        help='Generate synthetic scene-text images for language <lang>')
    parser.add_argument('--output_path', default='./',
                        help='path to store generated results')
    parser.add_argument('--total_samples', default=10000,
                        help='Total number of samples to generate')
    args = parser.parse_args()
    main(args.lang, args.output_path, args.total_samples, args.viz)

    cv2.destroyAllWindows()
    exit(0)