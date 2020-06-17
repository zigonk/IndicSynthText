# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

import pygame
from pygame import freetype
from text_utils import FontState
import numpy as np 
import matplotlib.pyplot as plt 
import _pickle as cPickle
import argparse


def gen_fontmodel(lang,output_path):
	pygame.init()


	ys = np.arange(8,200)
	A = np.c_[ys,np.ones_like(ys)]

	xs = []
	models = {} #linear model

	FS = FontState(lang,create_model=True)
	print(FS)
	#plt.figure()
	#plt.hold(True)
	for i in range(len(FS.fonts)):
		print(i)
		font = freetype.Font(FS.fonts[i], size=12)
		h = []
		for y in ys:
			y = float(y)
			h.append(font.get_sized_glyph_height(y))
		h = np.array(h)
		m,_,_,_ = np.linalg.lstsq(A,h)
		print(m)
		print(font.name)
		models[font.name] = m
		xs.append(h)

	with open(output_path+'/font_px2pt'+lang+'.cp','wb') as f:
		cPickle.dump(models,f)
	#plt.plot(xs,ys[i])
	#plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--lang',help='flag for turning on visualizations')
	parser.add_argument('--output_path',help='path to store font models')
	args = parser.parse_args()
	gen_fontmodel(args.lang,args.output_path)


