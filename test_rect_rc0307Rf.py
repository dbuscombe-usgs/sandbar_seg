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

   N_axis = np.linspace(minpts2[1], minpts2[1]+rows, rows)-2
   E_axis = np.linspace(minpts2[0], minpts2[0]+cols, cols)-2

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

## test!!

dat1 =  pickle.load( open( "RC0307Rf_homo_trans1.p", "rb" ) )
dat2 =  pickle.load( open( "RC0307Rf_homo_trans2.p", "rb" ) )
dat3 =  pickle.load( open( "RC0307Rf_homo_trans3.p", "rb" ) )
dat4 =  pickle.load( open( "RC0307Rf_homo_trans4.p", "rb" ) )
dat5 =  pickle.load( open( "RC0307Rf_homo_trans5.p", "rb" ) )
dat6 =  pickle.load( open( "RC0307Rf_homo_trans6.p", "rb" ) )
#dat7 =  pickle.load( open( "RC0307Rf_homo_trans7.p", "rb" ) )


##merged

impts = np.vstack((dat4['impts'], dat5['impts'], dat6['impts']))
mappts = np.vstack((dat4['mappts'], dat5['mappts'], dat6['mappts']))

pts2 = np.vstack((mappts[:,0],mappts[:,1])).T

# find minimum in e,n
minpts2 = np.min(pts2, axis=0)

# substract min e,n to put in metres from origin
pts2 =  pts2 - minpts2

homo = cv2.findHomography(impts, pts2)

dsize = tuple(np.ceil(np.max(pts2, axis=0)*2).astype('int'))

pickle.dump( {'trans':homo[0], 'dsize':dsize, 'impts':impts, 'mappts':mappts}, open( "RC0307Rf_homo_mergedtrans.p", "wb" ) )


infiles  = []
xyz = []

#1
infiles.append('RC0307Rf_surveydata/RC0307Rf_20091012_1530_reg.jpg')
xyz.append(np.genfromtxt('RC0307Rf_surveydata/30_091012_grid.TXT', delimiter=' '))

#2
infiles.append('RC0307Rf_surveydata/RC0307Rf_20111007_1350_reg.jpg')
xyz.append(np.genfromtxt('RC0307Rf_surveydata/30_111007_GRID.TXT', delimiter=' '))

#3
infiles.append('RC0307Rf_surveydata/RC0307Rf_20121004_1133_reg.jpg')
xyz.append(np.genfromtxt('RC0307Rf_surveydata/30_121004_GRID.TXT', delimiter=' '))

#4
infiles.append('RC0307Rf_surveydata/RC0307Rf_20121128_1356_reg.jpg')
xyz.append(np.genfromtxt('RC0307Rf_surveydata/30_121128_GRID.TXT', delimiter=' '))

#5
infiles.append('RC0307Rf_surveydata/RC0307Rf_20130923_1449.JPG_reg.jpg')
xyz.append(np.genfromtxt('RC0307Rf_surveydata/30_130923_GRID.TXT', delimiter=' '))

##6
infiles.append('RC0307Rf_surveydata/RC0307Rf_20140926_1327.JPG_reg.jpg')
xyz.append(np.genfromtxt('RC0307Rf_surveydata/30_140926_GRID.TXT', delimiter=' '))

#7
infiles.append('RC0307Rf_surveydata/RC0307Rf_20150925_1138.JPG_reg.jpg')
xyz.append(np.genfromtxt('RC0307Rf_surveydata/30_150925_GRID.TXT', delimiter=' '))




times = ['Oct 12, 2009, 15:30', 'Oct 7, 2011, 13:50', 'Oct 4, 2012, 11:33', 'Nov 28, 2012, 13:56', 'Sept 23, 2013, 14:49', 'Sept 26, 2014, 13:27', 'Sept 25, 2015, 11:38']

## rect 1, image 1 - 7
for k in range(7):
   img = cv2.imread(infiles[k])
   img = img[500:1500,:y,:d]

   N_axis, E_axis, dst = dotrans(img,homo[0],dsize,mappts)

   rectim = dst[:,:,0].astype('float32')
   rectim[rectim==0] = np.nan

   ax = plt.subplot(111)
   plt.imshow(bkimg, extent = imextent)
   plt.plot(mappts[:,0], mappts[:,1],'ro', markersize=2)
   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)
   plt.xlim(campos[1]-50, campos[1]+50)
   plt.ylim(campos[0]-75, campos[0]+125)

   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
   plt.title(times[k])
   ax.get_xaxis().get_major_formatter().set_useOffset(False)
   ax.get_yaxis().get_major_formatter().set_useOffset(False)
   labels = ax.get_xticklabels()
   plt.setp(labels, rotation=30, fontsize=10)
   labels = ax.get_yticklabels()
   plt.setp(labels, rotation=30, fontsize=10)

   #plt.show()
   plt.savefig('RC0307Rf_mergedrect_'+str(k)+'.png', bbox_inches = 'tight')
   plt.close()






### rect 1, image 1 - 7
#for k in range(7):
#   img = cv2.imread(infiles[k])
#   img = img[500:1500,:y,:d]

#   N_axis, E_axis, dst = dotrans(img,dat1['trans'],dat1['dsize'],dat1['mappts'])

#   rectim = dst[:,:,0].astype('float32')
#   rectim[rectim==0] = np.nan

#   plt.imshow(bkimg, extent = imextent)
#   plt.plot(dat1['mappts'][:,0], dat1['mappts'][:,1],'ro', markersize=2)
#   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)
#   plt.xlim(campos[1]-50, campos[1]+50)
#   plt.ylim(campos[0]-75, campos[0]+125)

#   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
#   #plt.show()
#   plt.savefig('RC0307Rf_rect1_'+str(k)+'.png', bbox_inches = 'tight')
#   plt.close()


### rect 2, image 1 - 7
#for k in range(7):
#   img = cv2.imread(infiles[k])
#   img = img[500:1500,:y,:d]

#   N_axis, E_axis, dst = dotrans(img,dat2['trans'],dat2['dsize'],dat2['mappts'])

#   rectim = dst[:,:,0].astype('float32')
#   rectim[rectim==0] = np.nan

#   plt.imshow(bkimg, extent = imextent)
#   plt.plot(dat2['mappts'][:,0], dat2['mappts'][:,1],'ro', markersize=2)
#   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)
#   plt.xlim(campos[1]-50, campos[1]+50)
#   plt.ylim(campos[0]-75, campos[0]+125)

#   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
#   #plt.show()
#   plt.savefig('RC0307Rf_rect2_'+str(k)+'.png', bbox_inches = 'tight')
#   plt.close()


### rect 3, image 1 - 7
#for k in range(7):
#   img = cv2.imread(infiles[k])
#   img = img[500:1500,:y,:d]

#   N_axis, E_axis, dst = dotrans(img,dat3['trans'],dat3['dsize'],dat3['mappts'])

#   rectim = dst[:,:,0].astype('float32')
#   rectim[rectim==0] = np.nan

#   plt.imshow(bkimg, extent = imextent)
#   plt.plot(dat3['mappts'][:,0], dat3['mappts'][:,1],'ro', markersize=2)
#   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)
#   plt.xlim(campos[1]-50, campos[1]+50)
#   plt.ylim(campos[0]-75, campos[0]+125)

#   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
#   #plt.show()
#   plt.savefig('RC0307Rf_rect3_'+str(k)+'.png', bbox_inches = 'tight')
#   plt.close()



### rect 4, image 1 - 7
#for k in range(7):
#   img = cv2.imread(infiles[k])
#   img = img[500:1500,:y,:d]

#   N_axis, E_axis, dst = dotrans(img,dat4['trans'],dat4['dsize'],dat4['mappts'])

#   rectim = dst[:,:,0].astype('float32')
#   rectim[rectim==0] = np.nan

#   plt.imshow(bkimg, extent = imextent)
#   plt.plot(dat4['mappts'][:,0], dat4['mappts'][:,1],'ro', markersize=2)
#   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)
#   plt.xlim(campos[1]-50, campos[1]+50)
#   plt.ylim(campos[0]-75, campos[0]+125)

#   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
#   #plt.show()
#   plt.savefig('RC0307Rf_rect4_'+str(k)+'.png', bbox_inches = 'tight')
#   plt.close()



### rect 5, image 1 - 7
#for k in range(7):
#   img = cv2.imread(infiles[k])
#   img = img[500:1500,:y,:d]

#   N_axis, E_axis, dst = dotrans(img,dat5['trans'],dat5['dsize'],dat5['mappts'])

#   rectim = dst[:,:,0].astype('float32')
#   rectim[rectim==0] = np.nan

#   plt.imshow(bkimg, extent = imextent)
#   plt.plot(dat5['mappts'][:,0], dat5['mappts'][:,1],'ro', markersize=2)
#   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)
#   plt.xlim(campos[1]-50, campos[1]+50)
#   plt.ylim(campos[0]-75, campos[0]+125)

#   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
#   #plt.show()
#   plt.savefig('RC0307Rf_rect5_'+str(k)+'.png', bbox_inches = 'tight')
#   plt.close()



### rect 6, image 1 - 7
#for k in range(7):
#   img = cv2.imread(infiles[k])
#   img = img[500:1500,:y,:d]

#   N_axis, E_axis, dst = dotrans(img,dat6['trans'],dat6['dsize'],dat6['mappts'])

#   rectim = dst[:,:,0].astype('float32')
#   rectim[rectim==0] = np.nan

#   plt.imshow(bkimg, extent = imextent)
#   plt.plot(dat6['mappts'][:,0], dat6['mappts'][:,1],'ro', markersize=2)
#   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)
#   plt.xlim(campos[1]-50, campos[1]+50)
#   plt.ylim(campos[0]-75, campos[0]+125)

#   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
#   #plt.show()
#   plt.savefig('RC0307Rf_rect6_'+str(k)+'.png', bbox_inches = 'tight')
#   plt.close()




### rect 7, image 1 - 7
#for k in range(7):
#   img = cv2.imread(infiles[k])
#   img = img[500:1500,:y,:d]

#   N_axis, E_axis, dst = dotrans(img,dat7['trans'],dat7['dsize'],dat7['mappts'])

#   rectim = dst[:,:,0].astype('float32')
#   rectim[rectim==0] = np.nan

#   plt.imshow(bkimg, extent = imextent)
#   plt.plot(dat7['mappts'][:,0], dat7['mappts'][:,1],'ro', markersize=2)
#   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)
#   plt.xlim(campos[1]-50, campos[1]+50)
#   plt.ylim(campos[0]-75, campos[0]+125)

#   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
#   #plt.show()
#   plt.savefig('RC0307Rf_rect7_'+str(k)+'.png', bbox_inches = 'tight')
#   plt.close()










