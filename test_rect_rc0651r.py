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

   N_axis = np.linspace(minpts2[1], minpts2[1]+rows, rows)-1.5
   E_axis = np.linspace(minpts2[0], minpts2[0]+cols, cols)-1.5

   return N_axis, E_axis, dst



cs2cs_args = "epsg:26949"
trans =  pyproj.Proj(init=cs2cs_args)

bkimg = imread('tifs/64_9.tif')
x,y,d = bkimg.shape
pos = np.genfromtxt('tifs/64_9.tfw')
imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]


#65 mile
campos = [571347.521, 222586.675]


## test!!

dat1 =  pickle.load( open( "RC0651R_surveydata/RC0651R_homo_trans1.p", "rb" ) )
#dat2 =  pickle.load( open( "RC0651R_homo_trans2.p", "rb" ) )
#dat3 =  pickle.load( open( "RC0651R_homo_trans3.p", "rb" ) )

##merged

#impts = np.vstack((dat1['impts'], dat2['impts'], dat3['impts']))
#mappts = np.vstack((dat1['mappts'], dat2['mappts'], dat3['mappts']))

#pts2 = np.vstack((mappts[:,0],mappts[:,1])).T

## find minimum in e,n
#minpts2 = np.min(pts2, axis=0)

## substract min e,n to put in metres from origin
#pts2 =  pts2 - minpts2

#homo = cv2.findHomography(impts, pts2)

#dsize = tuple(np.ceil(np.max(pts2, axis=0)).astype('int'))

#pickle.dump( {'trans':homo[0], 'dsize':dsize, 'impts':impts, 'mappts':mappts}, open( "RC0651R_homo_mergedtrans.p", "wb" ) )


infiles  = []
xyz = []

#1
infiles.append('RC0651R_surveydata/RC0651R_20121009_1219_reg.jpg')
xyz.append(np.genfromtxt('RC0651R_surveydata/65_121009_GRID.TXT', delimiter=' '))

#2
infiles.append('RC0651R_surveydata/RC0651R_20130927_1201_reg.jpg')
xyz.append(np.genfromtxt('RC0651R_surveydata/65_130927_GRID.TXT', delimiter=' '))

#3
infiles.append('RC0651R_surveydata/RC0651R_20140930_1202.JPG_reg.jpg')
xyz.append(np.genfromtxt('RC0651R_surveydata/65_140930_GRID.TXT', delimiter=' '))


times = ['Oct 9, 2012, 12:19', 'Sept 27, 2013, 12:01', 'Sept 30, 2014, 12:02']

## rect 1, image 1 - 3
for k in range(3):
   img = cv2.imread(infiles[k])
   img = img[1200:2000,:y,:d]

   #N_axis, E_axis, dst = dotrans(img,homo[0],dsize,mappts)
   N_axis, E_axis, dst = dotrans(img,dat1['trans'],dat1['dsize'],dat1['mappts'])

   rectim = dst[:,:,0].astype('float32')
   rectim[rectim==0] = np.nan

   ax = plt.subplot(111)
   plt.imshow(bkimg, extent = imextent)
   plt.plot(dat1['mappts'][:,0], dat1['mappts'][:,1],'ro', markersize=2)
   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)

   plt.xlim(campos[1]-300, campos[1]-100)
   plt.ylim(campos[0]-350, campos[0]-100)

   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
   plt.title(times[k])
   ax.get_xaxis().get_major_formatter().set_useOffset(False)
   ax.get_yaxis().get_major_formatter().set_useOffset(False)
   labels = ax.get_xticklabels()
   plt.setp(labels, rotation=30, fontsize=10)
   labels = ax.get_yticklabels()
   plt.setp(labels, rotation=30, fontsize=10)

   #plt.show()
   plt.savefig('RC0651R_mergedrect_'+str(k)+'.png', bbox_inches = 'tight')
   plt.close()







