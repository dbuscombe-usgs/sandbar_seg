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

bkimg = imread('tifs/64_9.tif')
x,y,d = bkimg.shape
pos = np.genfromtxt('tifs/64_9.tfw')
imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]


#65 mile
campos = [571347.521, 222586.675]


###1
infile = 'RC0651R_surveydata/RC0651R_20121009_1219_reg.jpg'
xyz = np.genfromtxt('RC0651R_surveydata/65_121009_GRID.TXT', delimiter=' ')

####2
infile = 'RC0651R_surveydata/RC0651R_20130927_1201_reg.jpg'
xyz = np.genfromtxt('RC0651R_surveydata/65_130927_GRID.TXT', delimiter=' ')

###3
infile = 'RC0651R_surveydata/RC0651R_20140930_1202_reg.jpg'
xyz = np.genfromtxt('RC0651R_surveydata/65_140930_GRID.TXT', delimiter=' ')



img = cv2.imread(infile)
img = img[1000:2000,:y,:d]



# plot gcps on obliue, then rectified, images
fig = plt.figure(frameon=False)
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(bkimg, extent = imextent)
plt.plot(campos[1], campos[0],'ro')
plt.plot(xyz[:,1],xyz[:,2],'.', markersize=2)

plt.xlim(campos[1]-300, campos[1]-100)
plt.ylim(campos[0]-350, campos[0]-100)

pts = plt.ginput(n=100, timeout=1000)
plt.close()

pts = np.asarray(pts)



pts1 = np.array([[  3.46927016e+02,   3.44823589e+02],
       [  4.75427419e+02,   5.58990927e+02],
       [  2.55523024e+03,   7.06528427e+02],
       [  3.00736129e+03,   6.49417137e+02],
       [  2.30298871e+03,   4.16212702e+02],
       [  2.22362054e+05,   5.71005374e+05],
       [  2.22402759e+05,   5.71066912e+05],
       [  2.22385772e+05,   5.71170438e+05],
       [  2.22384169e+05,   5.71188386e+05],
       [  2.22362695e+05,   5.71154733e+05]])

pts2 = np.array([[  2.99334274e+02,   3.28166129e+02],
       [  2.24587742e+03,   4.23351613e+02],
       [  2.69800847e+03,   6.23241129e+02],
       [  2.49811895e+03,   6.94630242e+02],
       [  2.92645363e+03,   6.66074597e+02],
       [  2.22360693e+05,   5.71000224e+05],
       [  2.22362012e+05,   5.71154496e+05],
       [  2.22372231e+05,   5.71168341e+05],
       [  2.22385746e+05,   5.71169989e+05],
       [  2.22383439e+05,   5.71189109e+05]])


pts3 = np.array([[  4.84945968e+02,   3.28166129e+02],
       [  2.20780323e+03,   4.94740726e+02],
       [  2.42672984e+03,   4.23351613e+02],
       [  2.67421210e+03,   7.04148790e+02],
       [  3.11682460e+03,   6.70833871e+02],
       [  2.22360034e+05,   5.71000883e+05],
       [  2.22366297e+05,   5.71150541e+05],
       [  2.22362012e+05,   5.71154826e+05],
       [  2.22385416e+05,   5.71169660e+05],
       [  2.22383439e+05,   5.71188449e+05]])


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

dst = cv2.warpPerspective(img,homo[0],tuple(np.ceil(np.max(pts2, axis=0)*1.5).astype('int')))

pickle.dump( {'trans':homo[0], 'dsize':tuple(np.ceil(np.max(pts2, axis=0)*1.5).astype('int')), 'impts':impts, 'mappts':mappts}, open( "RC0651R_homo_trans3.p", "wb" ) )

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
plt.xlim(campos[1]-300, campos[1]-100)
plt.ylim(campos[0]-350, campos[0]-100)


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
plt.xlim(campos[1]-300, campos[1]-100)
plt.ylim(campos[0]-350, campos[0]-100)

plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')#, alpha=0.5)
#plt.show()

plt.savefig('RC0651R_rect_image_ex3.png')
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
