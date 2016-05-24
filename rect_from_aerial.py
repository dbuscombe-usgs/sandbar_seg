import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pyproj
import numpy as np
#from scipy.misc import imread
import cv2

#me = 'filfy'
me = 'dbuscombe'

cs2cs_args = "epsg:26949"

##coords for 22mile
#lon = -111.760
#lat = 36.613
##infile = '22mile.JPG'
#infile = '/home/filfy/github_clones/processing/detect_clusters/sandbars/RC0220Ra_20130510_0948.jpg'

#xyz = np.genfromtxt('/home/filfy/github_clones/processing/detect_clusters/sandbars/022_130922_GRID.TXT', delimiter=',')


#coords for 30mile
lon = -111.8476
lat = 36.5159
#infile = '22mile.JPG'
infile = '/home/'+me+'/github_clones/processing/detect_clusters/sandbars/RC0307Rf_20121004_1333.jpg'

xyz = np.genfromtxt('/home/'+me+'/github_clones/processing/detect_clusters/sandbars/030_121004_GRID.TXT', delimiter='\t')


img = cv2.imread(infile)

trans =  pyproj.Proj(init=cs2cs_args)   
#lon, lat = trans(e,n, inverse=True)

fig = plt.figure(frameon=False)

plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
map = Basemap(projection='merc', epsg=cs2cs_args.split(':')[1], 
    resolution = 'h', #h #f
    llcrnrlon=np.min(lon)-0.001, llcrnrlat=np.min(lat)-0.001,
    urcrnrlon=np.max(lon)+0.001, urcrnrlat=np.max(lat)+0.001)

try:
   map.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='World_Imagery', xpixels=1000, ypixels=None, dpi=300)
except:
   map.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='ESRI_Imagery_World_2D', xpixels=3000, ypixels=None, dpi=600)
   
pts = plt.ginput(n=100, timeout=1000)  

plt.close()
   
pts = np.asarray(pts)

impts = pts[:len(pts)/2]

mappts = pts[len(pts)/2:]

gx,gy = map.projtran(mappts[:,0], mappts[:,1], inverse=True)

e, n = trans(gx, gy)

pts2 = np.vstack((e,n)).T

# find minimum in e,n
minpts2 = np.min(pts2, axis=0)

# substract min e,n to put in metres from origin
pts2 =  pts2 - minpts2

usepts = np.arange(len(e)) # use all points

# fuind the perspective transformation between the 2 planes
homo = cv2.findHomography(impts[usepts], pts2[usepts])

dst = cv2.warpPerspective(img,homo[0],tuple(np.ceil(np.max(pts2, axis=0)).astype('int')))


rows,cols,ch = dst.shape

N_axis = np.linspace(minpts2[1], minpts2[1]+rows, rows)         
E_axis = np.linspace(minpts2[0], minpts2[0]+cols, cols)

plt.figure()
plt.subplot(221)
plt.imshow(img)
plt.plot(impts[usepts,0], impts[usepts,1],'ro')
plt.axis('tight')

plt.subplot(223)
map = Basemap(projection='merc', epsg=cs2cs_args.split(':')[1], 
    resolution = 'h', #h #f
    llcrnrlon=np.min(lon)-0.002, llcrnrlat=np.min(lat)-0.002,
    urcrnrlon=np.max(lon)+0.002, urcrnrlat=np.max(lat)+0.002)

try:
   map.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='World_Imagery', xpixels=1000, ypixels=None, dpi=300)
except:
   map.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='ESRI_Imagery_World_2D', xpixels=3000, ypixels=None, dpi=600)
   
plt.plot(mappts[:,0], mappts[:,1],'ro')

plt.subplot(222, aspect=cols/rows)
plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')
plt.plot(xyz[:,0],xyz[:,1],'.')
plt.plot(minpts2[0]+pts2[usepts,0], minpts2[1]+pts2[usepts,1],'ro')
plt.axis('equal'); plt.axis('tight'); 

plt.subplot(224, aspect=cols/rows)
plt.pcolormesh(E_axis, N_axis, dst[:,:,0], cmap='gray')
plt.plot(minpts2[0]+pts2[usepts,0], minpts2[1]+pts2[usepts,1],'ro')
plt.axis('equal'); plt.axis('tight'); 
plt.show()







