import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
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

bkimg = imread(root+'tifs/RM30_7.tif')
x,y,d = bkimg.shape
pos = np.genfromtxt(root+'tifs/RM30_7.tfw')
imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]

##1
#infile = 'RC0307Rf_20091012_1530_reg.jpg'
#xyz = np.genfromtxt('30_091012_grid.TXT', delimiter=' ')

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


img = cv2.imread(infile)
#img = img[500:1500,:y,:d]


## plot gcps on oblique, then rectified, images
#fig = plt.figure(frameon=False)
#plt.subplot(121)
#plt.imshow(img)

#plt.subplot(122)
#plt.imshow(bkimg, extent = imextent)
#plt.plot(campos[1], campos[0],'ro')
#plt.plot(xyz[:,1],xyz[:,2],'.', markersize=2)

#plt.xlim(campos[1]-50, campos[1]+50)
#plt.ylim(campos[0]-75, campos[0]+125)

#pts = plt.ginput(n=100, timeout=1000)
#plt.close()

#pts = np.asarray(pts)


pts1 = np.array([[  1.58298603e+02,   5.58246304e+02],
       [  8.33881099e+02,   8.51977824e+02],
       [  3.08092723e+03,   9.10724128e+02],
       [  1.14964248e+03,   3.15917800e+02],
       [  2.77985242e+03,   6.46365760e+02],
       [  2.19514684e+05,   6.11702525e+05],
       [  2.19575534e+05,   6.11772906e+05],
       [  2.19593496e+05,   6.11857950e+05],
       [  2.19516883e+05,   6.11769241e+05],
       [  2.19566370e+05,   6.11851352e+05]])

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


pts = pts1

impts = pts[:len(pts)/2]

impts[:,1] = impts[:,1]+500

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









