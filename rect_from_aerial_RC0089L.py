
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

bkimg = imread('tifs/RM8_5.tif')
x,y,d = bkimg.shape
pos = np.genfromtxt('tifs/RM8_5.tfw')
imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]


#9 mile
campos = [639366.86, 235900.411]


# deployment 1

###1
#infile = 'RC0089L_surveydata/RC0089L_20121004_0959_reg.jpg'
#xyz = np.genfromtxt('RC0089L_surveydata/9_121003_GRID.TXT', delimiter=' ')

#####2
#infile = 'RC0089L_surveydata/RC0089L_20121127_1556_reg.jpg'
#xyz = np.genfromtxt('RC0089L_surveydata/9_121127_GRID.TXT', delimiter=' ')

##3
infile = 'RC0089L_surveydata/RC0089L_20130921_1553_reg.jpg'
xyz = np.genfromtxt('RC0089L_surveydata/9_130921_GRID.TXT', delimiter=' ')

## deployment 2
####4
#infile = 'RC0089L_surveydata/RC0089L_20140924_0742.JPG_res.jpg'
#xyz = np.genfromtxt('RC0089L_surveydata/9_140924_GRID.TXT', delimiter=' ')

## deployment 3
####5
#infile = 'RC0089L_surveydata/RC0089L_20150923_1748.JPG'
#xyz = np.genfromtxt('RC0089L_surveydata/9_150923_GRID.TXT', delimiter=' ')



img = cv2.imread(infile)
img = img[1200:2000,:y,:d]


# plot gcps on obliue, then rectified, images
fig = plt.figure(frameon=False)
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(bkimg, extent = imextent)
plt.plot(campos[1], campos[0],'ro')
plt.plot(xyz[:,1],xyz[:,2],'.', markersize=2)

plt.xlim(campos[1], campos[1]+150)
plt.ylim(campos[0]-300, campos[0])

pts = plt.ginput(n=100, timeout=1000)
plt.close()

pts = np.asarray(pts)



pts1 = np.array([[  1.54462500e+02,   6.54438306e+02],
       [  7.39320968e+02,   6.04450403e+02],
       [  1.72908145e+03,   3.64508468e+02],
       [  2.01401250e+03,   2.09545968e+02],
       [  2.39392056e+03,   7.95774194e+01],
       [  3.25371250e+03,   2.95895161e+01],
       [  2.04900403e+03,   3.49512097e+02],
       [  1.95402702e+03,   3.99500000e+02],
       [  2.71384315e+03,   4.04498790e+02],
       [  3.29370282e+03,   1.69555645e+02],
       [  2.36028885e+05,   6.39278569e+05],
       [  2.36020578e+05,   6.39258790e+05],
       [  2.36019391e+05,   6.39194708e+05],
       [  2.36018996e+05,   6.39167018e+05],
       [  2.36009106e+05,   6.39135373e+05],
       [  2.35974296e+05,   6.39067730e+05],
       [  2.36009106e+05,   6.39186401e+05],
       [  2.36003964e+05,   6.39201828e+05],
       [  2.35966385e+05,   6.39184028e+05],
       [  2.35965198e+05,   6.39074455e+05]])




pts2 = np.array([[  1.59461290e+02,   6.49439516e+02],
       [  7.44319758e+02,   5.84455242e+02],
       [  2.01401250e+03,   2.09545968e+02],
       [  2.42391331e+03,   7.45786290e+01],
       [  3.26371008e+03,   3.45883065e+01],
       [  1.72908145e+03,   3.59509677e+02],
       [  1.92403427e+03,   3.99500000e+02],
       [  2.88880081e+03,   3.59509677e+02],
       [  3.37368347e+03,   2.09545968e+02],
       [  2.36028885e+05,   6.39277778e+05],
       [  2.36020578e+05,   6.39260373e+05],
       [  2.36018996e+05,   6.39166623e+05],
       [  2.36009106e+05,   6.39134582e+05],
       [  2.35973901e+05,   6.39068917e+05],
       [  2.36019391e+05,   6.39193126e+05],
       [  2.35995657e+05,   6.39206575e+05],
       [  2.35962034e+05,   6.39170974e+05],
       [  2.35955309e+05,   6.39074455e+05]])




pts3 = np.array([[  1.54462500e+02,   6.49439516e+02],
       [  7.19325806e+02,   5.79456452e+02],
       [  1.72408266e+03,   3.59509677e+02],
       [  2.02401008e+03,   2.09545968e+02],
       [  2.40891694e+03,   7.45786290e+01],
       [  3.26870887e+03,   3.45883065e+01],
       [  2.27894839e+03,   3.44513306e+02],
       [  2.86380685e+03,   3.69507258e+02],
       [  3.29870161e+03,   1.64556855e+02],
       [  2.36028885e+05,   6.39277778e+05],
       [  2.36020182e+05,   6.39259186e+05],
       [  2.36019787e+05,   6.39193917e+05],
       [  2.36018204e+05,   6.39166227e+05],
       [  2.36009106e+05,   6.39134977e+05],
       [  2.35973110e+05,   6.39067730e+05],
       [  2.35989328e+05,   6.39182841e+05],
       [  2.35964011e+05,   6.39176116e+05],
       [  2.35964011e+05,   6.39069313e+05]])


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

pickle.dump( {'trans':homo[0], 'dsize':tuple(np.ceil(np.max(pts2, axis=0)).astype('int')), 'impts':impts, 'mappts':mappts}, open( "RC0089L_homo_trans3.p", "wb" ) )

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
plt.xlim(campos[1], campos[1]+150)
plt.ylim(campos[0]-300, campos[0])
plt.plot(xyz[:,1],xyz[:,2],'.')

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
plt.xlim(campos[1], campos[1]+150)
plt.ylim(campos[0]-300, campos[0])

plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')#, alpha=0.5)
#plt.show()

plt.savefig('RC0089L_rect_image_ex3.png')
plt.close()




