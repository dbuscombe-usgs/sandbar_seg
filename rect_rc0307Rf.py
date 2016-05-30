
from fnmatch import filter 
import cPickle as pickle
import numpy.ma as ma
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from scipy.misc import imread, imresize
import scipy.misc

import warnings
warnings.filterwarnings("ignore")


def nancrop(a, x, y):
   nans = a[:,:,0]==0 #np.isnan(a)
   nancols = np.all(nans, axis=0) # 
   nanrows = np.all(nans, axis=1) # 

   firstcol = nancols.argmin() # 
   firstrow = nanrows.argmin() #

   lastcol = len(nancols) - nancols[::-1].argmin() # 
   lastrow = len(nanrows) - nanrows[::-1].argmin() #

   return a[firstrow:lastrow,firstcol:lastcol,:], x[firstrow:lastrow], y[firstcol:lastcol]


def dotrans(img,trans,size,mappts):

   dst = cv2.warpPerspective(img,trans,size)

   pts2 = np.vstack((mappts[:,0],mappts[:,1])).T

   # find minimum in e,n
   minpts2 = np.min(pts2, axis=0)

   rows,cols,ch = dst.shape

   N_axis = np.linspace(minpts2[1], minpts2[1]+rows, rows)
   E_axis = np.linspace(minpts2[0], minpts2[0]+cols, cols)

   return N_axis, E_axis, dst


cs2cs_args = "epsg:26949"
trans =  pyproj.Proj(init=cs2cs_args)

#coords for 30mile
lon = -111.8476
lat = 36.5159
campos = trans(lon, lat)
campos = campos[1], campos[0]

bkimg = imread('tifs/RM30_7.tif')
x,y,d = bkimg.shape
pos = np.genfromtxt('tifs/RM30_7.tfw')
imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]



dat =  pickle.load( open( "RC0307Rf_surveydata/RC0307Rf_homo_mergedtrans.p", "rb" ) )

direc = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf_regis/'

filenames = sorted(filter(os.listdir(direc), '*.[Jj][Pp][Gg]'))

for file in filenames:

   img = imread(direc+file) #cv2.imread(file)
   x,y,d = img.shape
   img = img[500:1500,:y,:d]
   N_axis, E_axis, dst = dotrans(img,dat['trans'],dat['dsize'],dat['mappts'])

   rectim, N, E = nancrop(dst, N_axis, E_axis)

   outfile = direc+file.split('_reg.jpg')[0]+'_rect_im.png'
   outfile = outfile.replace('regis', 'rect')


   ax = plt.subplot(111)
   plt.imshow(bkimg, extent = imextent)
   plt.xlim(campos[1]-75, campos[1]+75)
   plt.ylim(campos[0]-100, campos[0]+125)

   plt.imshow(ma.masked_where(rectim==0,rectim), extent = [E.min(), E.max(), N.min(), N.max()], origin='lower' )

   ax.get_xaxis().get_major_formatter().set_useOffset(False)
   ax.get_yaxis().get_major_formatter().set_useOffset(False)
   labels = ax.get_xticklabels()
   plt.setp(labels, rotation=30, fontsize=10)
   labels = ax.get_yticklabels()
   plt.setp(labels, rotation=30, fontsize=10)
   plt.savefig(outfile, bbox_inches = 'tight')
   plt.close()


   outfile = direc+file.split('_reg.jpg')[0]+'_rect.tif'
   outfile = outfile.replace('regis', 'rect')

   scipy.misc.toimage(rectim).save(outfile)

   f = open(outfile.replace('tif', 'tfw'),'wb')
   f.write(str((N.max()-N.min())/ np.shape(rectim)[0])+'\n')
   f.write(str(0)+'\n')
   f.write(str(0)+'\n')
   f.write(str(-(E.max()-E.min())/ np.shape(rectim)[1])+'\n')
   f.write(str(E.min())+'\n')
   f.write(str(N.min())+'\n')
   f.close()






