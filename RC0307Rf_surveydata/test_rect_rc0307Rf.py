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

#root = '/home/dbuscombe/github_clones/sandbar_seg/'

#cs2cs_args = "epsg:26949"
#trans =  pyproj.Proj(init=cs2cs_args)

##coords for 30mile
#lon = -111.8476
#lat = 36.5159
#campos = trans(lon, lat)
#campos = campos[1], campos[0]

#bkimg = imread(root+'tifs/RM30_7.tif')
#x,y,d = bkimg.shape
#pos = np.genfromtxt(root+'tifs/RM30_7.tfw')
#imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]

#======================================
#======================================

# get a background image from the gcmrc server

cs2cs_args = "epsg:26949"
trans =  pyproj.Proj(init=cs2cs_args)

#coords for 30mile
lon = -111.8476
lat = 36.5159
campos = trans(lon, lat)
campos = campos[1], campos[0]

#from owslib.wms import WebMapService
#wms = WebMapService('http://grandcanyon.usgs.gov/arcgis/services/Imagery/ColoradoRiverImageryExplorer/MapServer/WmsServer?')

px_res = 0.05
xd = 150
yd = 150

#img = wms.getmap(   layers=['0'], srs='EPSG:26949', bbox=(campos[1]-xd, campos[0]-yd, campos[1]+xd, campos[0]+yd), size=((xd/px_res), (yd/px_res)), format='image/jpeg', transparent=False)
#out = open('test_en.jpg', 'wb')
#out.write(img.read())
#out.close()

bkimg = imread('test_en.jpg')
x,y,d = bkimg.shape
imextent = [campos[1]-xd, campos[1]+xd, campos[0]-yd,  campos[0]+yd]


#============================
# load the homography
dat =  pickle.load( open( "RC0307Rf_homo_trans1.p", "rb" ) )
T = dat['trans']

#============================

infiles  = []
xyz = []

#1
infiles.append('RC0307Rf_20091012_1530_reg.jpg')
xyz.append(np.genfromtxt('30_091012_grid.TXT', delimiter=' '))

#2
infiles.append('RC0307Rf_20111007_1350_reg.jpg')
xyz.append(np.genfromtxt('30_111007_GRID.TXT', delimiter=' '))

#3
infiles.append('RC0307Rf_20121004_1133_reg.jpg')
xyz.append(np.genfromtxt('30_121004_GRID.TXT', delimiter=' '))

#4
infiles.append('RC0307Rf_20121128_1356_reg.jpg')
xyz.append(np.genfromtxt('30_121128_GRID.TXT', delimiter=' '))

#5
infiles.append('RC0307Rf_20130923_1449.JPG_reg.jpg')
xyz.append(np.genfromtxt('30_130923_GRID.TXT', delimiter=' '))

##6
infiles.append('RC0307Rf_20140926_1327.JPG_reg.jpg')
xyz.append(np.genfromtxt('30_140926_GRID.TXT', delimiter=' '))

#7
infiles.append('RC0307Rf_20150925_1138.JPG_reg.jpg')
xyz.append(np.genfromtxt('30_150925_GRID.TXT', delimiter=' '))

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

   plt.imshow(rectim, origin='lower', extent = [E.min(), E.max(), N.min(), N.max()], cmap='gray' )
   #plt.plot(tx+E.min(), ty+N.min(), 'r', lw=2) 
   plt.plot(xyz[counter][:,1], xyz[counter][:,2],'k.')
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






##dat2 =  pickle.load( open( "RC0307Rf_homo_trans2.p", "rb" ) )
##dat3 =  pickle.load( open( "RC0307Rf_homo_trans3.p", "rb" ) )
##dat4 =  pickle.load( open( "RC0307Rf_homo_trans4.p", "rb" ) )
##dat5 =  pickle.load( open( "RC0307Rf_homo_trans5.p", "rb" ) )
##dat6 =  pickle.load( open( "RC0307Rf_homo_trans6.p", "rb" ) )
##dat7 =  pickle.load( open( "RC0307Rf_homo_trans7.p", "rb" ) )


###merged

#impts = np.vstack((dat4['impts'], dat5['impts'], dat6['impts']))
#mappts = np.vstack((dat4['mappts'], dat5['mappts'], dat6['mappts']))

#pts2 = np.vstack((mappts[:,0],mappts[:,1])).T

## find minimum in e,n
#minpts2 = np.min(pts2, axis=0)

## substract min e,n to put in metres from origin
#pts2 =  pts2 - minpts2

#homo = cv2.findHomography(impts, pts2)

#dsize = tuple(np.ceil(np.max(pts2, axis=0)).astype('int'))

#pickle.dump( {'trans':homo[0], 'dsize':dsize, 'impts':impts, 'mappts':mappts}, open( "RC0307Rf_homo_mergedtrans.p", "wb" ) )




#times = ['Oct 12, 2009, 15:30', 'Oct 7, 2011, 13:50', 'Oct 4, 2012, 11:33', 'Nov 28, 2012, 13:56', 'Sept 23, 2013, 14:49', 'Sept 26, 2014, 13:27', 'Sept 25, 2015, 11:38']

### rect 1, image 1 - 7
#for k in range(7):
#   img = cv2.imread(infiles[k])
#   #img = img[500:1500,:y,:d]

#   N_axis, E_axis, dst = dotrans(img,homo[0],dsize,mappts)

#   rectim = dst[:,:,0].astype('float32')
#   rectim[rectim==0] = np.nan

#   ax = plt.subplot(111)
#   plt.imshow(bkimg, extent = imextent)
#   plt.plot(mappts[:,0], mappts[:,1],'ro', markersize=2)
#   plt.plot(xyz[k][:,1],xyz[k][:,2],'.', markersize=2)
#   plt.xlim(campos[1]-50, campos[1]+50)
#   plt.ylim(campos[0]-75, campos[0]+125)

#   plt.pcolormesh(E_axis, N_axis, ma.masked_where(np.isnan(rectim),rectim), cmap='gray', alpha=0.75)
#   plt.title(times[k])
#   ax.get_xaxis().get_major_formatter().set_useOffset(False)
#   ax.get_yaxis().get_major_formatter().set_useOffset(False)
#   labels = ax.get_xticklabels()
#   plt.setp(labels, rotation=30, fontsize=10)
#   labels = ax.get_yticklabels()
#   plt.setp(labels, rotation=30, fontsize=10)

#   #plt.show()
#   plt.savefig('RC0307Rf_mergedrect_'+str(k)+'.png', bbox_inches = 'tight')
#   plt.close()






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










