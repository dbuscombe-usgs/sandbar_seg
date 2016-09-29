import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import pyproj
import numpy as np
from scipy.misc import imread, imresize
import cv2
import cPickle as pickle

import numpy.ma as ma


#======================================
#======================================
# subfunctions

#======================================
def nancrop(a, x, y):
   nans = a[:,:,0]==0 #np.isnan(a)
   nancols = np.all(nans, axis=0) # 
   nanrows = np.all(nans, axis=1) # 

   firstcol = nancols.argmin() # 
   firstrow = nanrows.argmin() #

   lastcol = len(nancols) - nancols[::-1].argmin() # 
   lastrow = len(nanrows) - nanrows[::-1].argmin() #

   return a[firstrow:lastrow,firstcol:lastcol,:], x[firstrow:lastrow], y[firstcol:lastcol]


#======================================
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

from owslib.wms import WebMapService
wms = WebMapService('http://grandcanyon.usgs.gov/arcgis/services/Imagery/ColoradoRiverImageryExplorer/MapServer/WmsServer?')


root = '/home/dbuscombe/github_clones/sandbar_seg/'

#coords for 9mile
campos = [639366.86, 235900.411]

barpos = [639140.000, 235966.000]


px_res = 0.1
xd = 100
yd = 200

#img = wms.getmap(   layers=['0'], srs='EPSG:26949', bbox=(barpos[1]-xd, barpos[0]-yd, barpos[1]+xd, barpos[0]+yd), size=((xd/px_res), (yd/px_res)), format='image/jpeg', transparent=False)
#out = open('test_en_9mile.jpg', 'wb')
#out.write(img.read())
#out.close()


bkimg = imread('test_en_9mile.jpg')
x,y,d = bkimg.shape
imextent = [barpos[1]-xd, barpos[1]+xd, barpos[0]-yd,  barpos[0]+yd]



#============================
# load the homography
dat =  pickle.load( open( "RC0089L_deployment1_homo_trans1.p", "rb" ) )
T = dat['trans']

#============================

infiles  = []
xyz = []

#1
infiles.append('RC0089L_20121004_0959_reg.jpg')
xyz.append(np.genfromtxt('9_121003_GRID.TXT', delimiter=' '))

#2
infiles.append('RC0089L_20121127_1556_reg.jpg')
xyz.append(np.genfromtxt('9_121127_GRID.TXT', delimiter=' '))

#3
infiles.append('RC0089L_20130921_1553_reg.jpg')
xyz.append(np.genfromtxt('9_130921_GRID.TXT', delimiter=' '))



counter=0
for infile in infiles:

   img = imread(infile) #imresize(, 0.25) 

   # transform the image
   N_axis, E_axis, dst = dotrans(img,T,dat['dsize'],dat['mappts'])

   # crop the transformed image
   rectim, N, E = nancrop(dst, N_axis, E_axis)

   # make plot
   fig = plt.figure()

   ax=plt.subplot(111)
   plt.imshow(bkimg, extent = imextent)
   plt.xlim(campos[1]-xd/2, campos[1]+xd/2)
   plt.ylim(campos[0]-yd/2, campos[0]+yd/2)

   plt.imshow(rectim, origin='lower', extent = [E.min(), E.max(), N.min(), N.max()], cmap='gray', alpha=0.5 )
   #plt.plot(tx+E.min(), ty+N.min(), 'r', lw=2) 
   plt.plot(xyz[counter][:,1], xyz[counter][:,2],'k.', markersize=2)
   plt.axis('image'); 

   ax.get_xaxis().get_major_formatter().set_useOffset(False)
   ax.get_yaxis().get_major_formatter().set_useOffset(False)
   labels = ax.get_xticklabels()
   plt.setp(labels, rotation=30, fontsize=8)
   labels = ax.get_yticklabels()
   plt.setp(labels, rotation=30, fontsize=8)
 
   plt.xlabel('Easting [m]', fontsize=8)
   plt.ylabel('Northing [m]', fontsize=8)

   #plt.show()
   plt.savefig(infile.split('.jpg')[0]+'_rect_image_xyz.png', bbox_inches = 'tight', dpi=600)
   plt.close()

   counter = counter+1







#cs2cs_args = "epsg:26949"
#trans =  pyproj.Proj(init=cs2cs_args)

#bkimg = imread('tifs/RM8_5.tif')
#x,y,d = bkimg.shape
#pos = np.genfromtxt('tifs/RM8_5.tfw')
#imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]



##9 mile
#campos = [639366.86, 235900.411]


### test!!

#dat1 =  pickle.load( open( "RC0089L_homo_trans1.p", "rb" ) )
#dat2 =  pickle.load( open( "RC0089L_homo_trans2.p", "rb" ) )
#dat3 =  pickle.load( open( "RC0089L_homo_trans3.p", "rb" ) )

###merged

#impts = np.vstack((dat1['impts'], dat2['impts'], dat3['impts']))
#mappts = np.vstack((dat1['mappts'], dat2['mappts'], dat3['mappts']))

#pts2 = np.vstack((mappts[:,0],mappts[:,1])).T

## find minimum in e,n
#minpts2 = np.min(pts2, axis=0)

## substract min e,n to put in metres from origin
#pts2 =  pts2 - minpts2

#homo = cv2.findHomography(impts, pts2)

#dsize = tuple(np.ceil(np.max(pts2, axis=0)).astype('int'))

#pickle.dump( {'trans':homo[0], 'dsize':dsize, 'impts':impts, 'mappts':mappts}, open( "RC0089L_homo_mergedtrans.p", "wb" ) )


#infiles  = []
#xyz = []



#times = ['Oct 4, 2012, 09:59', 'Nov 27, 2012, 15:56', 'Sept 21, 2013, 15:53']

### rect 1, image 1 - 3
#for k in range(3):
#   img = cv2.imread(infiles[k])
#   img = img[1200:2000,:y,:d]

#   N_axis, E_axis, dst = dotrans(img,homo[0],dsize,mappts)
#   #N_axis, E_axis, dst = dotrans(img,dat2['trans'],dat2['dsize'],dat2['mappts'])

#   rectim = dst[:,:,0].astype('float32')
#   rectim[rectim==0] = np.nan

#   ax = plt.subplot(111)
#   plt.imshow(bkimg, extent = imextent)
#   plt.plot(mappts[:,0], mappts[:,1],'ro', markersize=2)
#   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)

#   plt.xlim(campos[1], campos[1]+150)
#   plt.ylim(campos[0]-300, campos[0])

#   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
#   plt.title(times[k])
#   ax.get_xaxis().get_major_formatter().set_useOffset(False)
#   ax.get_yaxis().get_major_formatter().set_useOffset(False)
#   labels = ax.get_xticklabels()
#   plt.setp(labels, rotation=30, fontsize=10)
#   labels = ax.get_yticklabels()
#   plt.setp(labels, rotation=30, fontsize=10)

#   #plt.show()
#   plt.savefig('RC0089L_mergedrect_'+str(k)+'.png', bbox_inches = 'tight')
#   plt.close()







