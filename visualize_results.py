# Author: Ankush Gupta
# Date: 2015

from __future__ import division
import os
import lmdb
import six
import argparse
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def viz_textbb(text_im, charBB_list, wordBB, alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1)
    plt.imshow(text_im)
    H,W = text_im.shape[:2]

    # plot the character-BB:
    for i in range(len(charBB_list)):
        bbs = charBB_list[i]
        ni = bbs.shape[-1]
        for j in range(ni):
            bb = bbs[:,:,j]
            bb = np.c_[bb,bb[:,0]]
            plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)

    # plot the word-BB:
    for i in range(wordBB.shape[-1]):
        bb = wordBB[:,:,i]
        bb = np.c_[bb,bb[:,0]]
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        # visualize the indiv vertices:
        vcol = ['r','g','b','k']
        for j in range(4):
            plt.scatter(bb[0,j],bb[1,j],color=vcol[j])        

    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    plt.show(block=False)

def main(db_fname):
    cnt = 1
    env = lmdb.open(db_fname, map_size=1099511627776)
    with env.begin(write=True) as txn:
        t = txn.get('num-samples'.encode())
        t = t.decode()
        print ("total number of images : ", t)
        while True:
            label_key = 'label-%09d'.encode() % cnt
            img_key = 'image-%09d'.encode() % cnt
            imgbuf = txn.get(img_key)
            label = txn.get(label_key)
            if label == None or imgbuf == None:
                continue
            label = label.decode('utf-8')

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            plt.imshow(img)
            plt.show(block=False)
            if 'q' in input("next? ('q' to exit) : "):
                break
            cnt+=1

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_data_path', required=True, help='path to generated lmdb data')
    args = parser.parse_args()
    main(args.lmdb_data_path)

    

