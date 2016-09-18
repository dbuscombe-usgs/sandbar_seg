## has been developed at the Grand Canyon Monitoring & Research Center,
## U.S. Geological Survey
##
## Author: Daniel Buscombe
## Project homepage: <https://github.com/dbuscombe-usgs/sandbar_seg>
##

import matplotlib.pyplot as plt
import pyproj
import numpy as np
from scipy.misc import imread
import cv2
import cPickle as pickle

cs2cs_args = "epsg:26949"
trans =  pyproj.Proj(init=cs2cs_args)

root = '/home/dbuscombe/github_clones/sandbar_seg/'

#coords for 30mile
lon = -111.8476
lat = 36.5159
campos = trans(lon, lat)
campos = campos[1], campos[0]

from owslib.wms import WebMapService
wms = WebMapService('http://grandcanyon.usgs.gov/arcgis/services/Imagery/ColoradoRiverImageryExplorer/MapServer/WmsServer?')

#img = wms.getmap(   layers=['0'], srs='EPSG:4326', bbox=(lon-0.001, lat-0.001, lon+0.001, lat+0.001), size=(1000, 1000), format='image/jpeg', transparent=False)
#out = open('test.jpg', 'wb')
#out.write(img.read())
#out.close()


px_res = 0.05
xd = 150
yd = 150

img = wms.getmap(   layers=['0'], srs='EPSG:26949', bbox=(campos[1]-xd, campos[0]-yd, campos[1]+xd, campos[0]+yd), size=((xd/px_res), (yd/px_res)), format='image/jpeg', transparent=False)
out = open('test_en.jpg', 'wb')
out.write(img.read())
out.close()


bkimg = imread('test_en.jpg')
x,y,d = bkimg.shape
imextent = [campos[1]-xd, campos[1]+xd, campos[0]-yd,  campos[0]+yd]

##1
infile = 'RC0307Rf_20091012_1530_reg.jpg'
xyz = np.genfromtxt('30_091012_grid.TXT', delimiter=' ')


img = cv2.imread(infile)

## plot gcps on oblique, then rectified, images
#fig = plt.figure(frameon=False)
#plt.subplot(121)
#plt.imshow(img)

#plt.subplot(122)
#plt.imshow(bkimg, extent = imextent)
##plt.plot(campos[1], campos[0],'ro')
#plt.plot(xyz[:,1],xyz[:,2],'.', markersize=2)

#plt.xlim(campos[1]-50, campos[1]+100)
#plt.ylim(campos[0]-75, campos[0]+150)

#pts = plt.ginput(n=20, timeout=100)
#plt.close()

##pts = np.asarray(pts)


pts1 = np.array([[(424.88064516129043, 1045.4451612903231),
 (499.60967741935474, 1189.1548387096777),
 (827.26774193548363, 1361.6064516129036),
 (1741.2612903225804, 1056.9419354838715),
 (2310.3516129032255, 890.23870967741959),
 (2574.7774193548385, 1114.4258064516134),
 (2775.9709677419355, 1166.161290322581),
 (3080.6354838709672, 1419.0903225806455),
 (3833.6741935483865, 1591.5419354838714),
 (988.22258064516109, 735.03225806451655),
 (17.52903225806449, 1094.1758064516134),
 (219514.38910224204, 611715.9407383675),
 (219543.81217916514, 611731.51766144449),
 (219575.54294839592, 611771.61381529062),
 (219543.81217916514, 611804.78689221374),
 (219548.13910224204, 611835.65227682912),
 (219561.11987147282, 611842.28689221374),
 (219568.61987147282, 611854.979199906),
 (219592.8506407036, 611861.32535375212),
 (219608.13910224204, 611882.09458452137),
 (219505.7352560882, 611758.92150759825),
 (219509.92753558292, 611664.9489579706)]
])


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

pickle.dump( {'trans':homo[0], 'dsize':tuple(np.ceil(np.max(pts2, axis=0)*1.2).astype('int')), 'impts':impts, 'mappts':mappts}, open( "RC0307Rf_homo_trans1.p", "wb" ) )


plt.figure()
plt.subplot(221)
plt.imshow(img)
plt.plot(impts[usepts,0], impts[usepts,1],'ro')
plt.axis('tight')

plt.subplot(223)
plt.imshow(bkimg, extent = imextent)
plt.plot(mappts[:,0], mappts[:,1],'ro')
plt.xlim(campos[1]-50, campos[1]+50)
plt.ylim(campos[0]-75, campos[0]+125)

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
plt.xlim(campos[1]-50, campos[1]+50)
plt.ylim(campos[0]-75, campos[0]+125)

plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')#, alpha=0.5)
#plt.show()

plt.savefig('RC0307Rf_rect_image_ex1.png')
plt.close()



##2
#infile = 'RC0307Rf_surveydata/RC0307Rf_20111007_1350_reg.jpg'
#xyz = np.genfromtxt('RC0307Rf_surveydata/30_111007_GRID.TXT', delimiter=' ')

##3
#infile = 'RC0307Rf_surveydata/RC0307Rf_20121004_1133_reg.jpg'
#xyz = np.genfromtxt('RC0307Rf_surveydata/30_121004_GRID.TXT', delimiter=' ')

##4
#infile = 'RC0307Rf_surveydata/RC0307Rf_20121128_1356_reg.jpg'
#xyz = np.genfromtxt('RC0307Rf_surveydata/30_121128_GRID.TXT', delimiter=' ')

##5
#infile = 'RC0307Rf_surveydata/RC0307Rf_20130923_1449.JPG_reg.jpg'
#xyz = np.genfromtxt('RC0307Rf_surveydata/30_130923_GRID.TXT', delimiter=' ')

###6
#infile = 'RC0307Rf_surveydata/RC0307Rf_20140926_1327.JPG_reg.jpg'
#xyz = np.genfromtxt('RC0307Rf_surveydata/30_140926_GRID.TXT', delimiter=' ')

##7
#infile = 'RC0307Rf_20150925_1138.JPG_reg.jpg'
#xyz = np.genfromtxt('30_150925_GRID.TXT', delimiter=' ')


#pts2 = np.array([[  1.58298603e+02,   5.36216440e+02],
#       [  9.36687131e+02,   7.78544944e+02],
#       [  3.13233024e+03,   8.66664400e+02],
#       [  1.14964248e+03,   3.15917800e+02],
#       [  2.23644911e+03,   5.80276168e+02],
#       [  2.19514684e+05,   6.11702892e+05],
#       [  2.19573335e+05,   6.11779871e+05],
#       [  2.19593496e+05,   6.11859417e+05],
#       [  2.19516517e+05,   6.11769241e+05],
#       [  2.19558306e+05,   6.11831557e+05]])

#pts3 = np.array([[  1.89454032e+02,   5.46988508e+02],
#       [  6.79335484e+02,   7.26944960e+02],
#       [  1.28918790e+03,   8.61912298e+02],
#       [  2.16897500e+03,   6.96952218e+02],
#       [  2.79882258e+03,   6.76957056e+02],
#       [  1.12922661e+03,   3.92026008e+02],
#       [  2.19515409e+05,   6.11703040e+05],
#       [  2.19551050e+05,   6.11753040e+05],
#       [  2.19574383e+05,   6.11790732e+05],
#       [  2.19574896e+05,   6.11836630e+05],
#       [  2.19565922e+05,   6.11850732e+05],
#       [  2.19521819e+05,   6.11768938e+05]])

#pts4 = np.array([[  1.43612027e+02,   5.65589592e+02],
#       [  8.63254251e+02,   9.32753992e+02],
#       [  2.41268802e+03,   8.88694264e+02],
#       [  1.12761262e+03,   3.30604376e+02],
#       [  2.83125544e+03,   7.12455352e+02],
#       [  2.19515050e+05,   6.11702892e+05],
#       [  2.19573701e+05,   6.11773639e+05],
#       [  2.19586898e+05,   6.11840355e+05],
#       [  2.19516517e+05,   6.11769607e+05],
#       [  2.19567103e+05,   6.11851719e+05]])

#pts5 = np.array([[  9.22090110e+01,   5.43559728e+02],
#       [  8.48567675e+02,   8.96037552e+02],
#       [  2.19973267e+03,   6.90425488e+02],
#       [  3.07358394e+03,   9.18067416e+02],
#       [  1.14229920e+03,   3.23261088e+02],
#       [  2.19515050e+05,   6.11702892e+05],
#       [  2.19571869e+05,   6.11772540e+05],
#       [  2.19564171e+05,   6.11828258e+05],
#       [  2.19592396e+05,   6.11859050e+05],
#       [  2.19516883e+05,   6.11769974e+05]])

#pts6 = np.array([[  1.50955315e+02,   5.50903016e+02],
#       [  4.44686835e+02,   7.27141928e+02],
#       [  8.63254251e+02,   9.32753992e+02],
#       [  1.86928471e+03,   5.80276168e+02],
#       [  3.08092723e+03,   9.69470432e+02],
#       [  1.14964248e+03,   3.15917800e+02],
#       [  2.19514684e+05,   6.11702158e+05],
#       [  2.19545842e+05,   6.11735883e+05],
#       [  2.19575534e+05,   6.11771807e+05],
#       [  2.19549141e+05,   6.11812862e+05],
#       [  2.19593496e+05,   6.11859050e+05],
#       [  2.19516883e+05,   6.11769241e+05]])

#pts7 = np.array([[  1.50955315e+02,   5.58246304e+02],
#       [  1.14229920e+03,   3.23261088e+02],
#       [  2.23644911e+03,   6.09649320e+02],
#       [  3.11030038e+03,   9.32753992e+02],
#       [  2.19514317e+05,   6.11702158e+05],
#       [  2.19516883e+05,   6.11769607e+05],
#       [  2.19557939e+05,   6.11831557e+05],
#       [  2.19593130e+05,   6.11858683e+05]])






