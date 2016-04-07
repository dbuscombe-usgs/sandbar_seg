## has been developed at the Grand Canyon Monitoring & Research Center,
## U.S. Geological Survey
##
## Author: Daniel Buscombe
## Project homepage: <https://github.com/dbuscombe-usgs/PyHum>
##
##This software is in the public domain because it contains materials that originally came from 
##the United States Geological Survey, an agency of the United States Department of Interior. 
##For more information, see the official USGS copyright policy at 
##http://www.usgs.gov/visual-id/credit_usgs.html#copyright
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

'''
===============================================================================
Interactive Sandbar Segmentation using GrabCut algorithm.

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

from __future__ import print_function

import Tkinter
from Tix import *

import ttk
from tkFileDialog import askopenfilename
import os
from PIL import Image, ImageTk

#import webbrowser
import tkMessageBox
from ScrolledText import ScrolledText

from scipy.misc import imresize
import pickle
from scipy.signal import convolve2d
from skimage.morphology import remove_small_holes, dilation, disk
from skimage.measure import regionprops, label

import numpy as np
import cv2


##======================================================
def std_convoluted(image, N):
    """
    bbbbb
    """
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)

    kernel = np.ones((2*N+1, 2*N+1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")

    return s/ns , np.sqrt((s2 - s**2 / ns) / ns)

##======================================================
def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    BLUE = [255,0,0]        # rectangle color
    BLACK = [0,0,0]         # sure BG
    WHITE = [255,255,255]   # sure FG

    DRAW_BG = {'color' : BLACK, 'val' : 0}
    DRAW_FG = {'color' : WHITE, 'val' : 1}
    thickness = 2           # brush thickness

    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print(" Now press the key 'n' a few times until no further change \n")

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

#################################################
def gui():

        print(__doc__)

        #filename = '/home/dbuscombe/github_clones/sandbar_seg/30mile.JPG'

	#=======================
	# NOTE: Frame will make a top-level window if one doesn't already exist which
	# can then be accessed via the frame's master attribute
	# make a Frame whose parent is root, named "pyhum"
	master = Tkinter.Frame(name='sandbar segtool')

	self = master.master  # short-cut to top-level window
	master.pack()  # pack the Frame into root, defaults to side=TOP
	self.title('Sandbar Segmentation Tool')  # name the window
   	       
	# create notebook
	demoPanel = Tkinter.Frame(master, name='demo')  # create a new frame slaved to master
	demoPanel.pack()  # pack the Frame into root

	# create (notebook) demo panel
	nb = ttk.Notebook(demoPanel, name='notebook')  # create the ttk.Notebook widget

	# extend bindings to top level window allowing
	#   CTRL+TAB - cycles thru tabs
	#   SHIFT+CTRL+TAB - previous tab
	#   ALT+K - select tab using mnemonic (K = underlined letter)
	nb.enable_traversal()

	nb.pack(fill=Tkinter.BOTH, expand=Tkinter.Y, padx=2, pady=3)  # add margin

	#==============================================================
	#==============================================================
	#========START about tab

	# Populate the second pane. Note that the content doesn't really matter
	read_frame = Tkinter.Frame(nb)
	nb.add(read_frame, text='Read & Process Images')#, state='disabled')

	read_frame.configure(background='GoldenRod1')

	Read_msg = [
	    "Read a .DAT and associated set of .SON files recorded by a Humminbird(R) instrument.\n\n",
	    "Parse the data into a set of memory mapped files that will",
	    "subsequently be used by the other functions of the PyHum module.\n\n" ,  
	    "Export time-series data and metadata in other formats.\n\n",    
	    "Create a kml file for visualising boat track be selected.\n\n"
	    "Create rudimentary plots of the data"]

	#lbl2 = Tkinter.Label(read_frame, wraplength='4i', justify=Tkinter.LEFT, anchor=Tkinter.N, text=''.join(Read_msg))

	#lbl2.configure(background='thistle3', fg="black")
		        
	## position and set resize behavior
	#lbl2.grid(row=0, column=0, columnspan=1, sticky='new', pady=5)

	#def hello1():
	#   tkMessageBox.showinfo("Read Data Instructions", "DAT file: path to the .DAT file\n\n SON files: path to *.SON files\n\n Model: 3 or 4 number code indicating model of sidescan unit\n\n Bed Pick: 1 = auto, 0 = manual, 3=auto with manual override\n\n Sound speed: typically, 1450 m/s in freshwater, 1500 in saltwater\n\n Transducer length: m\n\n Draft: m\n\n cs2cs_args: argument given to pyproj to turn wgs84 coords. to projection supported by proj.4. Default='epsg:26949'\n\n Chunk: partition data to chunks. 'd' - distance, m. 'p' - number of pings. 'h' - change in heading, degs. '1' - just 1 chunk\n\n Flip Port/Star: flip port and starboard sidescans\n\n Filter bearing: spike removing filter to bearing\n\n Calculate bearing: recaluclate bearing from positions")

	def hello1_alt():
	   try:
	      root = Tk()
	      root.wm_title("Read module")
	      S = Scrollbar(root)
	      T = Text(root, height=40, width=60, wrap=WORD)
	      S.pack(side=RIGHT, fill=Y)
	      T.pack(side=LEFT, fill=Y)
	      S.config(command=T.yview)
	      T.config(yscrollcommand=S.set)

	      T.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
	      T.tag_configure('big', font=('Verdana', 20, 'bold'))
	      T.tag_configure('color', foreground='#476042', font=('Tempus Sans ITC', 12, 'bold'))

	      T.insert(END, __doc__)
	   except:
	      hello1()   

	MSG1_btn = Tkinter.Button(read_frame, text = "Instructions", command = hello1_alt)
	MSG1_btn.grid(row=0, column=1, pady=(2,4))
	MSG1_btn.configure(background='thistle3', fg="black")

	read_frame.rowconfigure(1, weight=1)
	read_frame.columnconfigure((0,1), weight=1, uniform=1)

	#=======================
	# get son files
	sonVar = Tkinter.StringVar()
	self.read_son_btn = Tkinter.Button(read_frame, text='Get image files', underline=0,
		         command=lambda v=sonVar: _get_SON(master, v))
	son = Tkinter.Label(read_frame, textvariable=sonVar, name='dat')
	self.read_son_btn.grid(row=1, column=1, pady=(2,4))
	self.read_son_btn.configure(background='thistle3', fg="black")

	#=======================
	# process button
	proc_btn = Tkinter.Button(read_frame, text='Process!', underline=0,
		         command=lambda :_proc(self))
	proc_btn.grid(row=7, column=1, pady=(2,4))
	proc_btn.configure(background='thistle3', fg="black")

	#=======================
	def _get_SON(master, v):
	    self.SONfiles = askopenfilename(filetypes=[("Image files","*.JPG")], multiple=True)

	    for k in xrange(len(self.SONfiles)):
	       print(self.SONfiles[k])
	    self.folder = os.path.dirname(self.SONfiles[0])
	    
	    #self.son_btn.configure(fg='thistle3', background="black")

	    self.read_son_btn.configure(fg='thistle3', background="black")
	    
	    self.update() 

	#=======================
	def _proc(self):

            global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

            for filename in self.SONfiles:
	       print('Processing ' + filename)

               BLUE = [255,0,0]        # rectangle color
               BLACK = [0,0,0]         # sure BG
               WHITE = [255,255,255]   # sure FG

               DRAW_BG = {'color' : BLACK, 'val' : 0}
               DRAW_FG = {'color' : WHITE, 'val' : 1}

               # setting up flags
               rect = (0,0,1,1)
               drawing = False         # flag for drawing curves
               rectangle = False       # flag for drawing rect
               rect_over = False       # flag to check if rect drawn
               rect_or_mask = 100      # flag for selecting rect or mask mode
               value = DRAW_FG         # drawing initialized to FG
               thickness = 2           # brush thickness

               Athres = 1000
               scale = 0.25

               img = imresize(cv2.imread(filename),scale) #resize image so quarter size
    
               imagehsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
               im = imresize(cv2.imread(filename,0),scale) #resize image so quarter size
    
               la = cv2.Laplacian(im,cv2.CV_64F)

               # get std and mean through stndard deviation, fast thru convolution
               m1, s1 = std_convoluted(im, .5)
               m2, s2 = std_convoluted(im, .25)    
    
               #img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over
               img2 = img.copy()                               # a copy of original image
               mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
               output = np.zeros(img.shape,np.uint8)           # output image to be shown

               # input and output windows
               cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)

               cv2.namedWindow('input', cv2.WINDOW_AUTOSIZE)

               cv2.setMouseCallback('input',onmouse)
               cv2.moveWindow('input',img.shape[1]+10,90)

               print(" Instructions: \n")
               print(" Draw a rectangle around the object using right mouse button \n")

               while(1):

                   cv2.imshow('output',output)
                   cv2.imshow('input',img)
                   k = 0xFF & cv2.waitKey(1)

                   # key bindings
                   if k == 27:         # esc to exit
                       break
                   elif k == ord('0'): # BG drawing
                       print(" mark background regions with left mouse button \n")
                       value = DRAW_BG
                   elif k == ord('1'): # FG drawing
                       print(" mark foreground regions with left mouse button \n")
                       value = DRAW_FG
                   elif k == ord('s'): # save image
                       bar = np.zeros((img.shape[0],5,3),np.uint8)
                       res = np.hstack((img2,bar,img,bar,output))
                       cv2.imwrite(filename+'_output.png',res)
                       print(" Result saved as image \n")

                       pickle.dump( {'image':img, 'mask':output}, open( filename+"_out.p", "wb" ) )

                   elif k == ord('r'): # reset everything
                       print("resetting \n")
                       rect = (0,0,1,1)
                       drawing = False
                       rectangle = False
                       rect_or_mask = 100
                       rect_over = False
                       value = DRAW_FG
                       img = img2.copy()
                       mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
                       output = np.zeros(img.shape,np.uint8)           # output image to be shown
                   elif k == ord('n'): # segment the image
                       print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                       and again press 'n' \n""")
                       if (rect_or_mask == 0):         # grabcut with rect
                           bgdmodel = np.zeros((1,65),np.float64)
                           fgdmodel = np.zeros((1,65),np.float64)

                           cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                
                           try:
                              mask[imagehsv[:,:,0]>np.percentile(imagehsv[:,:,0],75)] = 2
                              mask[imagehsv[:,:,1]>np.percentile(imagehsv[:,:,1],75)] = 2
                              mask[imagehsv[:,:,2]<np.percentile(imagehsv[:,:,2],50)] = 2
      
                              mask[s1>np.percentile(s1,75)] = 2
                              mask[s2>np.percentile(s2,75)] = 2
                              mask[la>np.percentile(la,75)] = 2

                              mask[m1<np.percentile(m1,50)] = 2
                              mask[m2<np.percentile(m2,50)] = 2
                           except:
                              pass
                           
                           rect_or_mask = 1
                       elif rect_or_mask == 1:         # grabcut with mask
                           bgdmodel = np.zeros((1,65),np.float64)
                           fgdmodel = np.zeros((1,65),np.float64)
                           cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)

                   mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')

                   try:
                      l = label(mask2)
                      for region in regionprops(l):
                         if (region.area<Athres): 
                            l[l==region.label] = 0
        
                      mask2 = (l>0).astype('uint8')
                 
                      mask2 = dilation(mask2, disk(3))
                      mask2 = remove_small_holes(mask2, min_size=10000).astype('uint8') 
                   except:
                      pass 
                        
                   output = cv2.bitwise_and(img2,img2,mask=mask2)

               cv2.destroyAllWindows()

	#==============================================================
	#==============================================================
	#========END about tab

	# start app
	master.mainloop()


# =========================================================
# =========================================================
if __name__ == '__main__':
   
   gui()

