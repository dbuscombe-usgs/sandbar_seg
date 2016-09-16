
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

   if np.abs(t0[0])>50:
       t0[0] = np.sign(t0[0])*50
   if np.abs(t0[1])>100:
       t0[1] = np.sign(t0[1])*100

   timg = ird.transform_img(im1, tvec=t0)
   outfile = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0450L_regis/'+file.split(os.sep)[-1].split('.jpg')[0]+'_reg.jpg'
   scipy.misc.toimage(timg).save(outfile)
   print(outfile+' saved')

master = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0450L/RC0450L_20080304_0756.JPG'

im0 = imread(master, flatten=True)

filenames = sorted(glob('/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0450L/*.jpg'))

for file in filenames:
   doproc(im0, file)

filenames = sorted(glob('/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0450L/*.JPG'))

for file in filenames:
   doproc(im0, file)

#Parallel(n_jobs = cpu_count(), verbose=0)(delayed(doproc)(im0, file) for file in filenames)

#filenames = sorted(glob('/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0450L/*.JPG'))

#Parallel(n_jobs = cpu_count(), verbose=0)(delayed(doproc)(im0, file) for file in filenames)
