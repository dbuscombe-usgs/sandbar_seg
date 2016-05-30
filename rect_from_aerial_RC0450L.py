import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import pyproj
import numpy as np
from scipy.misc import imread
import cv2
import cPickle as pickle


cs2cs_args = "epsg:26949"


trans =  pyproj.Proj(init=cs2cs_args)
#lon, lat = trans(e,n, inverse=True)

bkimg = imread('tifs/RM.tif')
x,y,d = bkimg.shape
pos = np.genfromtxt('tifs/RM.tfw')
imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]


36.38395081
-111.8583715
#45 mile
campos = [622474.549, 227515.424]

###1
infile = 'RC0450L_surveydata/RC0450L_20091014_1334.jpg'
xyz = np.genfromtxt('RC0450L_surveydata/45_091014_Grid.TXT', delimiter=' ')

####2

###3


img = cv2.imread(infile)
img = img[500:1500,:y,:d]



## plot gcps on obliue, then rectified, images
#fig = plt.figure(frameon=False)
#plt.subplot(121)
#plt.imshow(img)

#plt.subplot(122)
#plt.imshow(bkimg, extent = imextent)
#plt.plot(campos[1], campos[0],'ro')
#plt.plot(xyz[:,1],xyz[:,2],'.', markersize=2)

#plt.xlim(campos[1]-200, campos[1]+10)
#plt.ylim(campos[0]-100, campos[0]+150)

#pts = plt.ginput(n=100, timeout=1000)
#plt.close()

#pts = np.asarray(pts)

pts1 = np.array([[  6.44842742e+01,   7.84431048e+02],
       [  1.73907903e+03,   9.69386290e+02],
       [  2.48889758e+03,   7.89429839e+02],
       [  2.97877903e+03,   7.79432258e+02],
       [  4.39393548e+02,   4.74506048e+02],
       [  2.27355874e+05,   6.22468233e+05],
       [  2.27403343e+05,   6.22521635e+05],
       [  2.27382905e+05,   6.22556907e+05],
       [  2.27389827e+05,   6.22578004e+05],
       [  2.27334118e+05,   6.22483726e+05]])

pts2 = np.array([[  9.94758065e+01,   7.59437097e+02],
       [  1.67909355e+03,   9.24397177e+02],
       [  2.19396895e+03,   7.19446774e+02],
       [  2.99377540e+03,   7.69434677e+02],
       [  4.39393548e+02,   4.64508468e+02],
       [  2.27354885e+05,   6.22466585e+05],
       [  2.27401035e+05,   6.22520316e+05],
       [  2.27371038e+05,   6.22553281e+05],
       [  2.27390157e+05,   6.22578333e+05],
       [  2.27333129e+05,   6.22483726e+05]])

pts3 = np.array([[  1.24469758e+02,   7.24445565e+02],
       [  1.91903548e+03,   9.24397177e+02],
       [  2.98377782e+03,   8.29420161e+02],
       [  2.32393750e+03,   7.24445565e+02],
       [  4.64387500e+02,   4.64508468e+02],
       [  2.27353896e+05,   6.22469222e+05],
       [  2.27406969e+05,   6.22529217e+05],
       [  2.27399387e+05,   6.22567455e+05],
       [  2.27374005e+05,   6.22554599e+05],
       [  2.27333459e+05,   6.22483726e+05]])


pts = pts1.copy()


impts = pts[:len(pts)/2]
mappts = pts[len(pts)/2:]
pts2 = np.vstack((mappts[:,0],mappts[:,1])).T

# find minimum in e,n
minpts2 = np.min(pts2, axis=0)

# substract min e,n to put in metres from origin
pts2 =  pts2 - minpts2
#
usepts = np.arange(len(impts)) # use all points
#
# # fuind the perspective transformation between the 2 planes
homo = cv2.findHomography(impts[usepts], pts2[usepts])

dst = cv2.warpPerspective(img,homo[0],tuple(np.ceil(np.max(pts2, axis=0)*1.5).astype('int')))

pickle.dump( {'trans':homo[0], 'dsize':tuple(np.ceil(np.max(pts2, axis=0)*1.5).astype('int')), 'impts':impts, 'mappts':mappts}, open( "RC0220Ra_homo_trans1.p", "wb" ) )

rows,cols,ch = dst.shape

N_axis = np.linspace(minpts2[1], minpts2[1]+rows, rows)-1.5
E_axis = np.linspace(minpts2[0], minpts2[0]+cols, cols)-1.5

plt.figure()
plt.subplot(221)
plt.imshow(img)
plt.plot(impts[usepts,0], impts[usepts,1],'ro')
plt.axis('tight')

plt.subplot(223)
plt.imshow(bkimg, extent = imextent)
plt.plot(mappts[:,0], mappts[:,1],'ro')
plt.xlim(campos[1]-200, campos[1]-50)
plt.ylim(campos[0]-50, campos[0]+100)

plt.subplot(222, aspect=cols/rows)
plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')
plt.plot(xyz[:,1],xyz[:,2],'.')
plt.plot(minpts2[0]+pts2[usepts,0], minpts2[1]+pts2[usepts,1],'ro')
plt.axis('equal'); plt.axis('tight');

plt.subplot(224, aspect=cols/rows)
# plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')
# plt.plot(minpts2[0]+pts2[usepts,0], minpts2[1]+pts2[usepts,1],'ro')
# plt.axis('equal'); plt.axis('tight');
# plt.show()
plt.imshow(bkimg, extent = imextent)
plt.plot(mappts[:,0], mappts[:,1],'ro')
plt.xlim(campos[1]-200, campos[1]-50)
plt.ylim(campos[0]-50, campos[0]+100)
plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')#, alpha=0.5)
#plt.show()

plt.savefig('RC0220Ra_rect_image_ex1.png')
plt.close()


# map = Basemap(projection='merc', epsg=cs2cs_args.split(':')[1],
#     resolution = 'h', #h #f
#     llcrnrlon=np.min(lon)-0.001, llcrnrlat=np.min(lat)-0.001,
#     urcrnrlon=np.max(lon)+0.001, urcrnrlat=np.max(lat)+0.001)
#
# try:
#    map.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='World_Imagery', xpixels=1000, ypixels=None, dpi=300)
# except:
#    map.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='ESRI_Imagery_World_2D', xpixels=3000, ypixels=None, dpi=600)
#

# gx,gy = map.projtran(mappts[:,0], mappts[:,1], inverse=True)
#
# e, n = trans(gx, gy)
#
