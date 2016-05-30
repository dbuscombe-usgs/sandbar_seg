
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
infile = 'RC0089L_surveydata/RC0089L_20121004_0959_res.jpg'
xyz = np.genfromtxt('RC0089L_surveydata/9_121003_GRID.TXT', delimiter=' ')

####2
infile = 'RC0089L_surveydata/RC0089L_20121127_1556_res.jpg'
xyz = np.genfromtxt('RC0089L_surveydata/9_121127_GRID.TXT', delimiter=' ')

###3
infile = 'RC0089L_surveydata/RC0089L_20130921_1553_res.jpg'
xyz = np.genfromtxt('RC0089L_surveydata/9_130921_GRID.TXT', delimiter=' ')

###4
infile = 'RC0089L_surveydata/RC0089L_20140924_0742.JPG_res.jpg'
xyz = np.genfromtxt('RC0089L_surveydata/9_140924_GRID.TXT', delimiter=' ')

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



pts1 = np.array([[  1.69458871e+02,   6.44440726e+02],
       [  7.34322177e+02,   5.84455242e+02],
       [  1.73408024e+03,   3.54510887e+02],
       [  1.96402460e+03,   4.09497581e+02],
       [  2.71884194e+03,   4.14496371e+02],
       [  3.26371008e+03,   1.74554435e+02],
       [  3.25871129e+03,   3.45883065e+01],
       [  2.36028885e+05,   6.39278569e+05],
       [  2.36021369e+05,   6.39258790e+05],
       [  2.36020182e+05,   6.39193917e+05],
       [  2.36004755e+05,   6.39201433e+05],
       [  2.35965989e+05,   6.39186401e+05],
       [  2.35965989e+05,   6.39072873e+05],
       [  2.35973505e+05,   6.39067730e+05]])


pts2 = np.array([[  1.54462500e+02,   6.39441935e+02],
       [  7.29323387e+02,   5.84455242e+02],
       [  1.72908145e+03,   3.54510887e+02],
       [  1.91403669e+03,   3.99500000e+02],
       [  2.91379476e+03,   3.59509677e+02],
       [  3.37368347e+03,   1.94549597e+02],
       [  3.26371008e+03,   2.95895161e+01],
       [  2.36028885e+05,   6.39278964e+05],
       [  2.36021369e+05,   6.39259582e+05],
       [  2.36019391e+05,   6.39193917e+05],
       [  2.35996053e+05,   6.39206575e+05],
       [  2.35963220e+05,   6.39171369e+05],
       [  2.35954913e+05,   6.39074851e+05],
       [  2.35973110e+05,   6.39068521e+05]])


pts3 = np.array([[  1.59461290e+02,   6.34443145e+02],
       [  7.14327016e+02,   5.79456452e+02],
       [  1.72908145e+03,   3.44513306e+02],
       [  2.88380202e+03,   3.69507258e+02],
       [  3.29370282e+03,   1.64556855e+02],
       [  3.26870887e+03,   2.95895161e+01],
       [  2.36028885e+05,   6.39276987e+05],
       [  2.36020182e+05,   6.39257999e+05],
       [  2.36019391e+05,   6.39193917e+05],
       [  2.35963616e+05,   6.39176512e+05],
       [  2.35962429e+05,   6.39068917e+05],
       [  2.35971527e+05,   6.39067335e+05]])


pts4 = np.array([[  1.69458871e+02,   6.74433468e+02],
       [  7.44319758e+02,   6.29444355e+02],
       [  1.76407298e+03,   3.84503629e+02],
       [  3.05376089e+03,   4.29492742e+02],
       [  3.37868226e+03,   2.49536290e+02],
       [  2.36028885e+05,   6.39278569e+05],
       [  2.36020182e+05,   6.39259582e+05],
       [  2.36019787e+05,   6.39194313e+05],
       [  2.35962429e+05,   6.39172161e+05],
       [  2.35965989e+05,   6.39084740e+05]])


pts = pts4.copy()


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

pickle.dump( {'trans':homo[0], 'dsize':tuple(np.ceil(np.max(pts2, axis=0)).astype('int')), 'impts':impts, 'mappts':mappts}, open( "RC0089L_homo_trans4.p", "wb" ) )

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

plt.savefig('RC0089L_rect_image_ex4.png')
plt.close()




