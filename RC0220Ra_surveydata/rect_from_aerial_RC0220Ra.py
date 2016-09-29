
import matplotlib.pyplot as plt
import pyproj
import numpy as np
from scipy.misc import imread
import cv2
import cPickle as pickle

cs2cs_args = "epsg:26949"
trans =  pyproj.Proj(init=cs2cs_args)

root = '/home/dbuscombe/github_clones/sandbar_seg/'

from owslib.wms import WebMapService
wms = WebMapService('http://grandcanyon.usgs.gov/arcgis/services/Imagery/ColoradoRiverImageryExplorer/MapServer/WmsServer?')

#coords for 22mile
campos = [622474.549, 227515.424]

barpos = [622518.000, 227361.000]


px_res = 0.05
xd = 200
yd = 100

img = wms.getmap(   layers=['0'], srs='EPSG:26949', bbox=(barpos[1]-xd, barpos[0]-yd, barpos[1]+xd, barpos[0]+yd), size=((xd/px_res), (yd/px_res)), format='image/jpeg', transparent=False)
out = open('test_en_RC0220Ra.jpg', 'wb')
out.write(img.read())
out.close()


bkimg = imread('test_en_RC0220Ra.jpg')
x,y,d = bkimg.shape
imextent = [barpos[1]-xd, barpos[1]+xd, barpos[0]-yd,  barpos[0]+yd]


# deployment 7

###1
infile = 'RC0220Ra_20130922_1353_reg.jpg'
xyz = np.genfromtxt('22_130922_GRID.TXT', delimiter=' ')


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


pts1 = np.array([[  5.12354839e+01,   1.26388387e+03],
       [  4.47874194e+02,   9.82212903e+02],
       [  5.74338710e+02,   7.92516129e+02],
       [  3.51751290e+03,   1.28112903e+03],
       [  4.47874194e+02,   2.48829032e+03],
       [  1.78724839e+03,   1.45932903e+03],
       [  2.48855161e+03,   1.29837419e+03],
       [  2.27354011e+05,   6.22467615e+05],
       [  2.27335086e+05,   6.22484282e+05],
       [  2.27314978e+05,   6.22494026e+05],
       [  2.27398366e+05,   6.22600692e+05],
       [  2.27451000e+05,   6.22471974e+05],
       [  2.27403688e+05,   6.22521462e+05],
       [  2.27382989e+05,   6.22557103e+05]])


#22 mile
#campos = [622474.549, 227515.424]


####2
#infile = 'RC0220Ra_surveydata/RC0220Ra_20140925_1342.JPG_reg.jpg'
#xyz = np.genfromtxt('RC0220Ra_surveydata/22_140925_GRID.TXT', delimiter=' ')

###3
#infile = 'RC0220Ra_surveydata/RC0220Ra_20150924_1141.JPG_reg.jpg'
#xyz = np.genfromtxt('RC0220Ra_surveydata/22_150924_GRID.TXT', delimiter=' ')


#pts2 = np.array([[  4.44891129e+01,   7.54438306e+02],
#       [  1.64910081e+03,   9.34394758e+02],
#       [  2.16897500e+03,   7.14447984e+02],
#       [  4.29395968e+02,   4.74506048e+02],
#       [  2.27356204e+05,   6.22467244e+05],
#       [  2.27401365e+05,   6.22519657e+05],
#       [  2.27372027e+05,   6.22554599e+05],
#       [  2.27333129e+05,   6.22483726e+05]])

#pts3 = np.array([[  9.94758065e+01,   7.34443145e+02],
#       [  1.85405121e+03,   9.39393548e+02],
#       [  2.98377782e+03,   8.54414113e+02],
#       [  4.54389919e+02,   4.79504839e+02],
#       [  2.27355545e+05,   6.22469552e+05],
#       [  2.27406969e+05,   6.22528228e+05],
#       [  2.27399717e+05,   6.22568114e+05],
#       [  2.27333129e+05,   6.22484385e+05]])


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

pickle.dump( {'trans':homo[0], 'dsize':tuple(np.ceil(np.max(pts2, axis=0)*1.2).astype('int')), 'impts':impts, 'mappts':mappts}, open( "RC0220Ra_deployment7_homo_trans1.p", "wb" ) )


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

plt.savefig('RC0220Ra_deployment7_rect_image_ex1.png')
plt.close()














#pts = pts3.copy()


#impts = pts[:len(pts)/2]
#mappts = pts[len(pts)/2:]
#pts2 = np.vstack((mappts[:,0],mappts[:,1])).T

## find minimum in e,n
#minpts2 = np.min(pts2, axis=0)

## substract min e,n to put in metres from origin
#pts2 =  pts2 - minpts2
##
#usepts = np.arange(len(impts)) # use all points
##
## # fuind the perspective transformation between the 2 planes
#homo = cv2.findHomography(impts[usepts], pts2[usepts])

#dst = cv2.warpPerspective(img,homo[0],tuple(np.ceil(np.max(pts2, axis=0)*1.5).astype('int')))

#pickle.dump( {'trans':homo[0], 'dsize':tuple(np.ceil(np.max(pts2, axis=0)*1.5).astype('int')), 'impts':impts, 'mappts':mappts}, open( "RC0220Ra_homo_trans3.p", "wb" ) )

#rows,cols,ch = dst.shape

#N_axis = np.linspace(minpts2[1], minpts2[1]+rows, rows)-1.5
#E_axis = np.linspace(minpts2[0], minpts2[0]+cols, cols)-1.5

#plt.figure()
#plt.subplot(221)
#plt.imshow(img)
#plt.plot(impts[usepts,0], impts[usepts,1],'ro')
#plt.axis('tight')

#plt.subplot(223)
#plt.imshow(bkimg, extent = imextent)
#plt.plot(mappts[:,0], mappts[:,1],'ro')
#plt.xlim(campos[1]-200, campos[1]-50)
#plt.ylim(campos[0]-50, campos[0]+100)

#plt.subplot(222, aspect=cols/rows)
#plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')
#plt.plot(xyz[:,1],xyz[:,2],'.')
#plt.plot(minpts2[0]+pts2[usepts,0], minpts2[1]+pts2[usepts,1],'ro')
#plt.axis('equal'); plt.axis('tight');

#plt.subplot(224, aspect=cols/rows)
## plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')
## plt.plot(minpts2[0]+pts2[usepts,0], minpts2[1]+pts2[usepts,1],'ro')
## plt.axis('equal'); plt.axis('tight');
## plt.show()
#plt.imshow(bkimg, extent = imextent)
#plt.plot(mappts[:,0], mappts[:,1],'ro')
#plt.xlim(campos[1]-200, campos[1]-50)
#plt.ylim(campos[0]-50, campos[0]+100)
#plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')#, alpha=0.5)
##plt.show()

#plt.savefig('RC0220Ra_rect_image_ex3.png')
#plt.close()


## map = Basemap(projection='merc', epsg=cs2cs_args.split(':')[1],
##     resolution = 'h', #h #f
##     llcrnrlon=np.min(lon)-0.001, llcrnrlat=np.min(lat)-0.001,
##     urcrnrlon=np.max(lon)+0.001, urcrnrlat=np.max(lat)+0.001)
##
## try:
##    map.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='World_Imagery', xpixels=1000, ypixels=None, dpi=300)
## except:
##    map.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='ESRI_Imagery_World_2D', xpixels=3000, ypixels=None, dpi=600)
##

## gx,gy = map.projtran(mappts[:,0], mappts[:,1], inverse=True)
##
## e, n = trans(gx, gy)
#
