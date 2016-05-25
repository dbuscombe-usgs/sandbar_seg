

from scipy.misc import imread
import matplotlib.pyplot as plt
import imreg_dft as ird
from glob import glob
from joblib import Parallel, delayed, cpu_count
import scipy.misc
import os

def doproc(im0, file):
   im1 = imread(file, flatten=True)
   t0, t1 = ird.translation(im0, im1)
   im1 = imread(file)
   timg = ird.transform_img(im1, tvec=t0)
   scipy.misc.toimage(timg).save('/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf_regis/'+file.split(os.sep)[-1].split('.jpg')[0]+'_reg.jpg')

master = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf/RC0307Rf_20091012_1130.jpg'

im0 = imread(master, flatten=True)

filenames = sorted(glob('/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf/*.jpg'))

Parallel(n_jobs = cpu_count(), verbose=1)(delayed(doproc)(im0, file) for file in filenames)
