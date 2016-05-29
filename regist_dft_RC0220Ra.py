
from scipy.misc import imread
import matplotlib.pyplot as plt
import imreg_dft as ird
from glob import glob
from joblib import Parallel, delayed, cpu_count
import scipy.misc
import os
import numpy as np

def doproc(im0, file):
   im1 = imread(file, flatten=True)

   t0, t1 = ird.translation(im0, im1)
   im1 = imread(file)

   # if np.abs(t0[0])>50:
   #    to[0] = np.sign(t0[0])*50
   # if np.abs(t0[1])>100:
   #    to[1] = np.sign(t0[1])*100

   timg = ird.transform_img(im1, tvec=t0)
   outfile = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0220Ra_regis/'+file.split(os.sep)[-1].split('.jpg')[0]+'_reg.jpg'
   scipy.misc.toimage(timg).save(outfile)
   print(outfile+' saved')

master = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0220Ra/RC0220Ra_20130922_1353.jpg'

im0 = imread(master, flatten=True)

filenames = sorted(glob('/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0220Ra/*.jpg'))

filenames = filenames[657:]

Parallel(n_jobs = cpu_count(), verbose=0)(delayed(doproc)(im0, file) for file in filenames)
