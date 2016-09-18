## has been developed at the Grand Canyon Monitoring & Research Center,
## U.S. Geological Survey
##
## Author: Daniel Buscombe
## Project homepage: <https://github.com/dbuscombe-usgs/sandbar_seg>
##
from __future__ import division
import cPickle as pickle
import numpy.ma as ma
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from scipy.misc import imread, imresize
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings("ignore")

from Tkinter import Tk
from tkFileDialog import askopenfilename, askdirectory

#======================================
#======================================
# subfunctions

#======================================
def nancrop(a, x, y):
   nans = a[:,:,0]==0 #np.isnan(a)
   nancols = np.all(nans, axis=0) # 
   nanrows = np.all(nans, axis=1) # 

   firstcol = nancols.argmin() # 
   firstrow = nanrows.argmin() #

   lastcol = len(nancols) - nancols[::-1].argmin() # 
   lastrow = len(nanrows) - nanrows[::-1].argmin() #

   return a[firstrow:lastrow,firstcol:lastcol,:], x[firstrow:lastrow], y[firstcol:lastcol]


#======================================
def dotrans(img,trans,size,mappts):

   dst = cv2.warpPerspective(img,trans,size)

   pts2 = np.vstack((mappts[:,0],mappts[:,1])).T

   # find minimum in e,n
   minpts2 = np.min(pts2, axis=0)

   rows,cols,ch = dst.shape

   N_axis = np.linspace(minpts2[1], minpts2[1]+rows, rows)
   E_axis = np.linspace(minpts2[0], minpts2[0]+cols, cols)

   return N_axis, E_axis, dst


#======================================
#======================================

# get a background image from the gcmrc server

cs2cs_args = "epsg:26949"
trans =  pyproj.Proj(init=cs2cs_args)

#coords for 30mile
lon = -111.8476
lat = 36.5159
campos = trans(lon, lat)
campos = campos[1], campos[0]

#from owslib.wms import WebMapService
#wms = WebMapService('http://grandcanyon.usgs.gov/arcgis/services/Imagery/ColoradoRiverImageryExplorer/MapServer/WmsServer?')

px_res = 0.05
xd = 150
yd = 150

#img = wms.getmap(   layers=['0'], srs='EPSG:26949', bbox=(campos[1]-xd, campos[0]-yd, campos[1]+xd, campos[0]+yd), size=((xd/px_res), (yd/px_res)), format='image/jpeg', transparent=False)
#out = open('test_en.jpg', 'wb')
#out.write(img.read())
#out.close()

bkimg = imread('test_en.jpg')
x,y,d = bkimg.shape
imextent = [campos[1]-xd, campos[1]+xd, campos[0]-yd,  campos[0]+yd]

#============================
# load the homography
dat =  pickle.load( open( "RC0307Rf_homo_trans1.p", "rb" ) )
T = dat['trans']

#============================
# load the output pickled file from the snadbar_segmentation gui

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
infiles = askopenfilename(filetypes=[("sandbar segmentation file","*.p")], multiple=True)

for infile in infiles:

   with open(infile, 'rb') as f:                                                                                    
      segdat = pickle.load(f)

   fct = 1/segdat['scale']

   img = imresize(segdat['img'], fct) 

   # transform the image
   N_axis, E_axis, dst = dotrans(img,T,dat['dsize'],dat['mappts'])

   # crop the transformed image
   rectim, N, E = nancrop(dst, N_axis, E_axis)

   # transform the sandbar outline
   x = segdat['contour_x']
   y = segdat['contour_y']

   pts = np.vstack((fct*x, fct*y, np.ones(len(x))))

   txyz = T.dot(pts)
   tx = txyz[0,:]
   ty = txyz[1,:]
   tz = txyz[2,:]

   tx = tx/tz 
   ty = ty/tz

   #=====================================
   #create mask
   nimg = Image.new('L', (rectim.shape[1], rectim.shape[0]), 0)
   polygon = zip(tx, ty)
   ImageDraw.Draw(nimg).polygon(polygon, outline=1, fill=1)
   mask = np.array(nimg)

   # maske greyscale image
   im = (0.299 * rectim[:,:,0] + 0.5870*rectim[:,:,1] + 0.114*rectim[:,:,2]).astype('uint8')

   # make plot
   fig = plt.figure()
   fig.subplots_adjust(wspace = 0.3)

   ax = plt.subplot(121)
   plt.imshow(img) #dat['out_img'])
   plt.plot(fct*segdat['contour_x'], fct*segdat['contour_y'],'r', lw=2)
   plt.axis('image'); 

   ax.get_xaxis().get_major_formatter().set_useOffset(False)
   ax.get_yaxis().get_major_formatter().set_useOffset(False)
   labels = ax.get_xticklabels()
   plt.setp(labels, rotation=30, fontsize=8)
   labels = ax.get_yticklabels()
   plt.setp(labels, rotation=30, fontsize=8)
 
   plt.xlabel('X [px]', fontsize=8)
   plt.ylabel('Y [px]', fontsize=8)

   ax = plt.subplot(122)
   plt.imshow(bkimg, extent = imextent)
   plt.xlim(campos[1]-xd/2, campos[1]+xd/2)
   plt.ylim(campos[0]-yd/2, campos[0]+yd/2)

   plt.imshow(ma.masked_where(mask==0,im), origin='lower', extent = [E.min(), E.max(), N.min(), N.max()], cmap='gray' )
   plt.plot(tx+E.min(), ty+N.min(), 'r', lw=2) 
   plt.axis('image'); 

   ax.get_xaxis().get_major_formatter().set_useOffset(False)
   ax.get_yaxis().get_major_formatter().set_useOffset(False)
   labels = ax.get_xticklabels()
   plt.setp(labels, rotation=30, fontsize=8)
   labels = ax.get_yticklabels()
   plt.setp(labels, rotation=30, fontsize=8)
 
   plt.xlabel('Easting [m]', fontsize=8)
   plt.ylabel('Northing [m]', fontsize=8)

   #plt.show()
   plt.savefig(infile.split('.jpg')[0]+'_rect_image_seg.png', bbox_inches = 'tight', dpi=600)
   plt.close()






