

import matplotlib.pyplot as plt
import pyproj
import numpy as np
from scipy.misc import imread
import cv2
import cPickle as pickle

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

img = wms.getmap(   layers=['0'], srs='EPSG:26949', bbox=(barpos[1]-xd, barpos[0]-yd, barpos[1]+xd, barpos[0]+yd), size=((xd/px_res), (yd/px_res)), format='image/jpeg', transparent=False)
out = open('test_en_9mile.jpg', 'wb')
out.write(img.read())
out.close()


bkimg = imread('test_en_9mile.jpg')
x,y,d = bkimg.shape
imextent = [barpos[1]-xd, barpos[1]+xd, barpos[0]-yd,  barpos[0]+yd]



# deployment 1

###1
infile = 'RC0089L_20121004_0959_reg.jpg'
xyz = np.genfromtxt('9_121003_GRID.TXT', delimiter=' ')

img = cv2.imread(infile)

# plot gcps on oblique, then rectified, images
fig = plt.figure(frameon=False)
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(bkimg, extent = imextent)
plt.plot(campos[1], campos[0],'ro')
plt.plot(xyz[:,1],xyz[:,2],'.', markersize=2)

plt.axis('tight')

pts = plt.ginput(n=14, timeout=100)
plt.close()

pts = np.asarray(pts)


#####2
#infile = 'RC0089L_20121127_1556_reg.jpg'
#xyz = np.genfromtxt('9_121127_GRID.TXT', delimiter=' ')

###3
#infile = 'RC0089L_20130921_1553_reg.jpg'
#xyz = np.genfromtxt('9_130921_GRID.TXT', delimiter=' ')

## deployment 2
####4
#infile = 'RC0089L_surveydata/RC0089L_20140924_0742.JPG_res.jpg'
#xyz = np.genfromtxt('RC0089L_surveydata/9_140924_GRID.TXT', delimiter=' ')

## deployment 3
####5
#infile = 'RC0089L_surveydata/RC0089L_20150923_1748.JPG'
#xyz = np.genfromtxt('RC0089L_surveydata/9_150923_GRID.TXT', delimiter=' ')


pts1 = np.array([[  1.60454839e+02,   1.85021935e+03],
       [  7.46790323e+02,   1.78698710e+03],
       [  1.72401613e+03,   1.56280000e+03],
       [  1.99419032e+03,   1.60303871e+03],
       [  2.72423548e+03,   1.60878710e+03],
       [  3.27033226e+03,   1.36735484e+03],
       [  3.49451935e+03,   1.33286452e+03],
       [  2.36029038e+05,   6.39279573e+05],
       [  2.36021054e+05,   6.39260419e+05],
       [  2.36019280e+05,   6.39194201e+05],
       [  2.36004495e+05,   6.39201315e+05],
       [  2.35967237e+05,   6.39187634e+05],
       [  2.35966054e+05,   6.39073804e+05],
       [  2.35959844e+05,   6.39057386e+05]])


#pts2 = 


#pts3 = 


pts = pts1
pts = np.squeeze(pts)

impts = pts[:len(pts)/2]

impts[:,1] = impts[:,1] #+500

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

dst = cv2.warpPerspective(img,homo[0],tuple(np.ceil(np.max(pts2, axis=0)*1.2).astype('int')))

rows,cols,ch = dst.shape

N_axis = np.linspace(minpts2[1], minpts2[1]+rows, rows)
E_axis = np.linspace(minpts2[0], minpts2[0]+cols, cols)

pickle.dump( {'trans':homo[0], 'dsize':tuple(np.ceil(np.max(pts2, axis=0)*1.2).astype('int')), 'impts':impts, 'mappts':mappts}, open( "RC0089L_deployment1_homo_trans1.p", "wb" ) )


plt.figure()
plt.subplot(221)
plt.imshow(img)
plt.plot(impts[usepts,0], impts[usepts,1],'ro')
plt.axis('tight')

plt.subplot(223)
plt.imshow(bkimg, extent = imextent)
plt.plot(mappts[:,0], mappts[:,1],'ro')
#plt.xlim(campos[1]-50, campos[1]+50)
#plt.ylim(campos[0]-75, campos[0]+125)

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
plt.plot(xyz[:,1],xyz[:,2],'.')
#plt.xlim(campos[1]-50, campos[1]+50)
#plt.ylim(campos[0]-75, campos[0]+125)

plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray') #, alpha=0.5)
#plt.axis('tight')
#plt.show()

plt.savefig('RC0089L_deployment1_rect_image_ex1.png')
plt.close()







