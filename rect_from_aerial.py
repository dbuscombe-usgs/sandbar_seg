import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import pyproj
import numpy as np
from scipy.misc import imread
import cv2
import cPickle as pickle


#me = 'filfy'
me = 'dbuscombe'

cs2cs_args = "epsg:26949"

#coords for 22mile
# lon = -111.760
# lat = 36.613
#infile = '22mile.JPG'
# infile = '/home/'+me+'/github_clones/processing/detect_clusters/sandbars/RC0220Ra_20130510_0948.jpg'
infile = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0220Ra/RC0220Ra_20130922_1553.jpg'
xyz = np.genfromtxt('/home/'+me+'/github_clones/processing/detect_clusters/sandbars/022_130922_GRID.TXT', delimiter=',')
#22 mile
campos = [622474.549, 227515.424]


# #coords for 30mile
# lon = -111.8476
# lat = 36.5159
# #infile = '22mile.JPG'
# infile = '/home/'+me+'/github_clones/processing/detect_clusters/sandbars/RC0307Rf_20121004_1333.jpg'
# infile = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf_regis/RC0307Rf_20091012_1130_reg.jpg'

# xyz = np.genfromtxt('/home/'+me+'/github_clones/processing/detect_clusters/sandbars/030_121004_GRID.TXT', delimiter='\t')

img = cv2.imread(infile)

trans =  pyproj.Proj(init=cs2cs_args)
#lon, lat = trans(e,n, inverse=True)

bkimg = imread('tifs/RM21_8.tif')
x,y,d = bkimg.shape
pos = np.genfromtxt('tifs/RM21_8.tfw')
imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]

img = img[500:1500,:y,:d]

# plot gcps on obliue, then rectified, images
fig = plt.figure(frameon=False)
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(bkimg, extent = imextent)
plt.plot(campos[1], campos[0],'ro')
plt.plot(xyz[:,0],xyz[:,1],'.')

plt.xlim(campos[1]-200, campos[1]+10)
plt.ylim(campos[0]-100, campos[0]+150)

pts = plt.ginput(n=100, timeout=1000)
plt.close()

pts = np.asarray(pts)

# pts = array([[  4.49391129e+02,   4.82004234e+02],
#        [  3.44915323e+01,   7.66935282e+02],
#        [  1.70908629e+03,   9.36894153e+02],
#        [  2.39392056e+03,   7.76932863e+02],
#        [  2.99877419e+03,   7.76932863e+02],
#        [  2.60386976e+03,   6.46964315e+02],
#        [  2.27333303e+05,   6.22484325e+05],
#        [  2.27356700e+05,   6.22467658e+05],
#        [  2.27401572e+05,   6.22519581e+05],
#        [  2.27385546e+05,   6.22559325e+05],
#        [  2.27390675e+05,   6.22578235e+05],
#        [  2.27374969e+05,   6.22569261e+05]])

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

pickle.dump( {'trans':homo[0], 'dsize':tuple(np.ceil(np.max(pts2, axis=0)).astype('int')), 'impts':impts, 'mappts':pts2}, open( "RC0220Ra_homo_trans.p", "wb" ) )

rows,cols,ch = dst.shape

N_axis = np.linspace(minpts2[1], minpts2[1]+rows, rows)
E_axis = np.linspace(minpts2[0], minpts2[0]+cols, cols)

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
plt.plot(xyz[:,0],xyz[:,1],'.')
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
plt.show()



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
