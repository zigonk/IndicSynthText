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
import io
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import h5py
import lmdb
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
# SECS_PER_IMG = 5 #max time per image in seconds
SECS_PER_IMG = None #max time per image in seconds
INSTANCE_PER_IMAGE = 900 # no. of times to use the same image
# path to the data-file, containing image, depth and segmentation:
DATA_PATH = './SynthTextGen/'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')

#@azhar
def filter_text(lang,text):
  unicode_range = {'odia':'[^\u0020-\u0040-\u0B00-\u0B7F]','kanada':'[^\u0020-\u0040-\u0C80-\u0CFF]',
  'tamil':'[^\u0020-\u0040-\u0B80-\u0BFF]','malyalam':'[^\u0020-\u0040-\u0D00-\u0D7F]',
  'urdu':'[^\u0020-\u0040-\u0600-\u06FF]','telgu':'[^\u0020-\u0040-\u0C00-\u0C7F]',
  'marathi':'[^\u0020-\u0040-\u0900-\u097F]','sanskrit':'[^\u0020-\u0040-\u0900-\u097F]',
  'hindi':'[^\u0020-\u0040-\u0900-\u097F]','ban':'[^\u0020-\u0040-\u0980-\u09FF]'}
  import re
  t = re.sub(unicode_range[lang],'',text)
  if len(text) == len(t):
    return False
  else:
    return True

#@azhar
def writeCache(env, cache):
  with env.begin(write=True) as txn:
    for k, v in cache.items():
      txn.put(k, v)

#@azhar
def checkImageIsValid(imageBin):
  if imageBin is None:
    return False
  imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
  #print('print imageBuf')
  #print(imageBuf,len(imageBuf))
  img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
  #print('print img after decoding')
  #print(img)

  imgH, imgW = img.shape[0], img.shape[1]
  if imgH * imgW == 0:
    return False
  return True

#@azhar
def crop(img,bbox):
  bbox = np.transpose(bbox)
  topleft_x = int(np.min(bbox[:,0]))
  topleft_y = int(np.min(bbox[:,1]))
  bot_right_x = int(np.max(bbox[:,0]))
  bot_right_y = int(np.max(bbox[:,1]))
  cropped_img = img[topleft_y:bot_right_y, topleft_x:bot_right_x]
  #plt.imshow(cropped_img)
  #plt.show()
  return cropped_img

def get_data():
  """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
  if not osp.exists(DB_FNAME):
    try:
      colorprint(Color.BLUE,'\tdownloading data (56 M) from: '+DATA_URL,bold=True)
      print()
      sys.stdout.flush()
      out_fname = 'data.tar.gz'
      wget.download(DATA_URL,out=out_fname)
      tar = tarfile.open(out_fname)
      tar.extractall()
      tar.close()
      os.remove(out_fname)
      colorprint(Color.BLUE,'\n\tdata saved at:'+DB_FNAME,bold=True)
      sys.stdout.flush()
    except:
      print (colorize(Color.RED,'Data not found and have problems downloading.',bold=True))
      sys.stdout.flush()
      sys.exit(-1)
  # open the h5 file and return:
  return h5py.File(DB_FNAME,'r')


def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in range(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']        
    #db['data'][dname].attrs['txt'] = res[i]['txt']
    L = res[i]['txt']
    print(L)
    L = [n.encode("UTF-8", "ignore") for n in L]
    print('in adding to data base')
    print(L)
    db['data'][dname].attrs['txt'] = L


def main(lang,out_path,total_samples,viz=False):
  # open databases:
  print (colorize(Color.BLUE,'getting data..',bold=True))
  #db = get_data()
  if osp.exists(DB_FNAME):
    db = h5py.File(DB_FNAME,'r')
  else:
    print(colorize(Color.RED,f'Data not found at {DB_FNAME}. Download from https://www.kaggle.com/azharshaikh/synthtextgen',bold=True))
    sys.stdout.flush()
    sys.exit(-1)

  print(colorize(Color.BLUE,'\t-> done',bold=True))

  # open the output h5 file:
  #out_db = h5py.File(OUT_FILE,'w')
  #out_db.create_group('/data')
  out_path = out_path+lang+'/'
  os.makedirs(out_path, exist_ok=True)
  env = lmdb.open(out_path, map_size=1099511627776)
  cache = {}
  cnt = 1

  print (colorize(Color.GREEN,'Storing the output in: '+out_path, bold=True))

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(DATA_PATH,lang,max_time=SECS_PER_IMG)
  for i in range(start_idx,end_idx):
    imname = imnames[i]
    try:
      # get the image:
      img = Image.fromarray(db['image'][imname][:])
      # get the pre-computed depth:
      #  there are 2 estimates of depth (represented as 2 "channels")
      #  here we are using the second one (in some cases it might be
      #  useful to use the other one):
      depth = db['depth'][imname][:].T
      depth = depth[:,:,1]
      # get segmentation:
      seg = db['seg'][imname][:].astype('float32')
      area = db['seg'][imname].attrs['area']
      label = db['seg'][imname].attrs['label']
      #print(label)

      # re-size uniformly:
      sz = depth.shape[:2][::-1]
      img = np.array(img.resize(sz,Image.ANTIALIAS))
      seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

      print (colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True))
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
      #print(res)
      if len(res) > 0:
        #@azhar
        for instance in range(len(res)):
          img = res[instance]['img']
          nw = len(res[instance]['txt'])
          #print('number of words',nw)
          for i in range(nw):
            label = res[instance]['txt'][i]
            if filter_text(lang,label):
              print('invalid word encountered')
              print(label)
              continue
            bbox = res[instance]['wordBB'][:,:,i]
            #print('bbox shape',bbox.shape)
            cropped_img = crop(img,bbox)
            #print('cropped image shape',cropped_img.shape)
            imgbin = cropped_img.tobytes()
            #print('image type ',type(imgbin))
            try:
              image_pil = Image.fromarray(cropped_img)
            except ValueError:
              continue
            imgByteArr = io.BytesIO()
            image_pil.save(imgByteArr, format='PNG')
            imgByteArr = imgByteArr.getvalue()

            if not checkImageIsValid(imgByteArr):
              print('%s is not a valid image' % label)
              continue
            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = imgByteArr
            cache[labelKey] = label.encode()
            if cnt % 1000 == 0:
              writeCache(env, cache)
              cache = {}
              print('Written %d samples' % (cnt))
            cnt += 1
            if cnt==total_samples:
              sys.exit()



          #imgbin = img.tobytes()
          #label = res[i]
        '''plt.imshow(res[0]['img'])
        plt.show()
        print(res[0]['wordBB'][:,:,0])
        print(np.shape(res[0]['wordBB'][:,:,0]))
        print(res[0]['txt'][0])
        crop(res[0]['img'],res[0]['wordBB'][:,:,0])'''
        #add_res_to_db()
        
        
        
        
        # non-empty : successful in placing text:
        #add_res_to_db(imname,res,out_db)
      # visualize the output:
      if viz:
        if 'q' in input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print (colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
      break
  cache['num-samples'.encode()] = str(cnt-1).encode()
  writeCache(env, cache)
  print('total samples:',cnt-1)
  db.close()
  #out_db.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Generate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  parser.add_argument('--lang',dest='lang',required=True, help='Generate synthetic scene-text images for language <lang>')
  parser.add_argument('--output_path',default='./',help='path to store generated results')
  parser.add_argument('--total_samples',default=10000,help='Total number of samples to generate')
  args = parser.parse_args()
  main(args.lang,args.output_path,args.total_samples,args.viz)
