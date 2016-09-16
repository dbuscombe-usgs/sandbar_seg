import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import pyproj
import numpy as np
from scipy.misc import imread
import cv2
import cPickle as pickle

import numpy.ma as ma


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

bkimg = imread('tifs/Seg_E_04mb.tif')
x,y,d = bkimg.shape
pos = np.genfromtxt('tifs/Seg_E_04mb.tfw')
imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]


#buckfarm
campos = [599027.225,	216585.344]


## test!!
dat1 =  pickle.load( open( "RC0414R_surveydata/RC0414R_homo_trans1.p", "rb" ) )
dat2 =  pickle.load( open( "RC0414R_surveydata/RC0414R_homo_trans2.p", "rb" ) )
dat3 =  pickle.load( open( "RC0414R_surveydata/RC0414R_homo_trans3.p", "rb" ) )


#merged

impts = np.vstack((dat1['impts'], dat2['impts'], dat3['impts']))
mappts = np.vstack((dat1['mappts'], dat2['mappts'], dat3['mappts']))

pts2 = np.vstack((mappts[:,0],mappts[:,1])).T

# find minimum in e,n
minpts2 = np.min(pts2, axis=0)

# substract min e,n to put in metres from origin
pts2 =  pts2 - minpts2

homo = cv2.findHomography(impts, pts2)

dsize = tuple(np.ceil(np.max(pts2, axis=0)).astype('int'))

pickle.dump( {'trans':homo[0], 'dsize':dsize, 'impts':impts, 'mappts':mappts}, open( "RC0414R_homo_mergedtrans.p", "wb" ) )


infiles  = []
xyz = []

#1
infiles.append('RC0414R_surveydata/RC0414R_20130924_1400.JPG')
xyz.append(np.genfromtxt('RC0414R_surveydata/41_130924_GRID.TXT', delimiter=' '))

#2
infiles.append('RC0414R_surveydata/RC0414R_20140927_1159.JPG')
xyz.append(np.genfromtxt('RC0414R_surveydata/41_140927_GRID.TXT', delimiter=' '))

#3
infiles.append('RC0414R_surveydata/RC0414R_20150926_0747.JPG')
xyz.append(np.genfromtxt('RC0414R_surveydata/41_150926_GRID.TXT', delimiter=' '))


times = ['Oct 24, 2013, 14:00', 'Sept 27, 2014, 11:59', 'Sept 26, 2015, 07:47']

## rect 1, image 1 - 3
for k in range(3):
   img = cv2.imread(infiles[k])
   img = img[1250:2500,:y,:d]

   N_axis, E_axis, dst = dotrans(img,homo[0],dsize,mappts)
   #N_axis, E_axis, dst = dotrans(img,dat1['trans'],dat1['dsize'],dat1['mappts'])

   rectim = dst[:,:,0].astype('float32')
   rectim[rectim==0] = np.nan

   ax = plt.subplot(111)
   plt.imshow(bkimg, extent = imextent)
   plt.plot(mappts[:,0], mappts[:,1],'ro', markersize=2)
   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)

   plt.xlim(campos[1]-300, campos[1]-75)
   plt.ylim(campos[0]-100, campos[0]+200)

   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
   plt.title(times[k])
   ax.get_xaxis().get_major_formatter().set_useOffset(False)
   ax.get_yaxis().get_major_formatter().set_useOffset(False)
   labels = ax.get_xticklabels()
   plt.setp(labels, rotation=30, fontsize=10)
   labels = ax.get_yticklabels()
   plt.setp(labels, rotation=30, fontsize=10)

   #plt.show()
   plt.savefig('RC0414R_mergedrect_'+str(k)+'.png', bbox_inches = 'tight')
   plt.close()













