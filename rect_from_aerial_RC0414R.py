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


bkimg = imread('tifs/Seg_E_04mb.tif')
x,y,d = bkimg.shape
pos = np.genfromtxt('tifs/Seg_E_04mb.tfw')
imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]


#buckfarm
campos = [599027.225,	216585.344]

###1
#infile = 'RC0414R_surveydata/RC0414R_20130924_1400.JPG'
#xyz = np.genfromtxt('RC0414R_surveydata/41_130924_GRID.TXT', delimiter=' ')

####2
#infile = 'RC0414R_surveydata/RC0414R_20140927_1159.JPG'
#xyz = np.genfromtxt('RC0414R_surveydata/41_140927_GRID.TXT', delimiter=' ')

###3
infile = 'RC0414R_surveydata/RC0414R_20150926_0747.JPG'
xyz = np.genfromtxt('RC0414R_surveydata/41_150926_GRID.TXT', delimiter=' ')


img = cv2.imread(infile)
img = img[1250:2500,:y,:d]


# plot gcps on obliue, then rectified, images
fig = plt.figure(frameon=False)
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(bkimg, extent = imextent)
plt.plot(campos[1], campos[0],'ro')
plt.plot(xyz[:,1],xyz[:,2],'.', markersize=2)

plt.xlim(campos[1]-300, campos[1]-75)
plt.ylim(campos[0]-100, campos[0]+200)

pts = plt.ginput(n=100, timeout=1000)
plt.close()

pts = np.asarray(pts)

pts1 = np.array([[  6.74188205e+02,   2.25521201e+02],
       [  1.56248063e+03,   1.95409594e+02],
       [  3.42940029e+03,   4.21246650e+02],
       [  2.76694492e+03,   4.58886159e+02],
       [  2.66908220e+03,   8.27753350e+02],
       [  1.86359670e+03,   6.92251117e+02],
       [  4.93723499e+01,   6.47083706e+02],
       [  2.16348308e+05,   5.99055231e+05],
       [  2.16367553e+05,   5.99110216e+05],
       [  2.16452230e+05,   5.99188295e+05],
       [  2.16425837e+05,   5.99154754e+05],
       [  2.16482472e+05,   5.99106367e+05],
       [  2.16441233e+05,   5.99080524e+05],
       [  2.16338410e+05,   5.98959007e+05]])


pts2 = np.array([[  6.96771911e+02,   2.10465397e+02],
       [  1.54742482e+03,   1.72825888e+02],
       [  3.36917707e+03,   4.43830355e+02],
       [  2.63897059e+03,   4.51358257e+02],
       [  1.99909893e+03,   6.99779019e+02],
       [  2.97793112e+02,   6.09444196e+02],
       [  2.16347758e+05,   5.99057430e+05],
       [  2.16366453e+05,   5.99109116e+05],
       [  2.16452230e+05,   5.99184996e+05],
       [  2.16428037e+05,   5.99165202e+05],
       [  2.16452230e+05,   5.99090421e+05],
       [  2.16335661e+05,   5.98962306e+05]])


pts3 = np.array([[  6.81716107e+02,   1.95409594e+02],
       [  1.53989692e+03,   1.57770084e+02],
       [  3.36917707e+03,   4.36302453e+02],
       [  2.25504760e+03,   6.62139509e+02],
       [  2.14986191e+02,   5.56748883e+02],
       [  2.16348308e+05,   5.99056880e+05],
       [  2.16365903e+05,   5.99109116e+05],
       [  2.16451680e+05,   5.99183897e+05],
       [  2.16495119e+05,   5.99110216e+05],
       [  2.16336211e+05,   5.98961206e+05]])


pts = pts3.copy()


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

dst = cv2.warpPerspective(img,homo[0],tuple(np.ceil(np.max(pts2, axis=0)).astype('int')))

pickle.dump( {'trans':homo[0], 'dsize':tuple(np.ceil(np.max(pts2, axis=0)*1.5).astype('int')), 'impts':impts, 'mappts':mappts}, open( "RC0414R_homo_trans3.p", "wb" ) )

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
plt.xlim(campos[1]-300, campos[1]-75)
plt.ylim(campos[0]-100, campos[0]+200)

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
plt.xlim(campos[1]-300, campos[1]-75)
plt.ylim(campos[0]-100, campos[0]+200)

plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')#, alpha=0.5)
#plt.show()

plt.savefig('RC0414R_rect_image_ex3.png')
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
