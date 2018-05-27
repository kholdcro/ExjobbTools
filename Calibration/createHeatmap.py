import sys, time
import numpy as np
import cv2
# assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import glob
from optparse import OptionParser
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def main():
  start = time.time()
  parser = OptionParser()  
  parser.add_option("-f", "--folder", dest="folder",
                  help="Choose I/O Folder", metavar="FILE")
  (options, args) = parser.parse_args()

  img = cv2.imread(options.folder+'/orbMatches.png')
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  xi = np.linspace(img.shape[0],0,img.shape[0])
  yi = np.linspace(img.shape[1],0,img.shape[1])
  img[0,0] = 0.
  print(np.max(img))
  print(np.unravel_index(np.argmax(img, axis=None), img.shape))
  values = np.nonzero(img)
  print(values)
  z = img[values]
  print(z)
  print(np.max(z))
  print(np.unravel_index(np.argmax(z, axis=None), z.shape))
  # z = np.log(z)
  zi = griddata((values[0], values[1]), z, (xi[None,:], yi[:,None]), method='cubic')
  print(np.unravel_index(np.argmax(zi, axis=None), zi.shape))

  # plotThings(xi,yi,zi.T,1,"Heatmap",options)
  plotOverlay(xi,yi,zi.T,1,"Heatmap",options)
  plotOverlay2(values,2,"ScatterMap",options)
  cv2.waitKey(0)

  input("Press [enter] to continue.")
  print("\nTotal Time:\t" + str(round(time.time()-start,3)))


def plotThings(x,y,z,figNum,name,opt):
  fig = plt.figure(figNum)

  CS = plt.contour(x,y,z,15,linewidths=0.5,colors='k')
  CS = plt.contourf(x,y,z,15,cmap=plt.cm.jet)
  plt.colorbar() # draw colorbar
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title(name)
  plt.grid(False)

  fig = plt.gcf()
  ax = fig.gca()
  plt.savefig(opt.folder + name+".png")
  plt.show(block = False)


def plotOverlay(x,y,z,figNum,name,opt):
  img = cv2.imread(opt.folder+'images/image0.jpg')
  mycmap = transparent_cmap(plt.cm.jet)

  ind = np.unravel_index(np.argmax(z, axis=None), z.shape)

  fig, ax = plt.subplots(1, 1)
  ax.imshow(img)
  ax.scatter(ind[0],ind[1], color="g", s=100)

  CS = plt.contour(x,y,z,15,linewidths=0.5,colors='k')
  CS = plt.contourf(x,y,z,15,cmap=mycmap)
  plt.colorbar(CS) # draw colorbar
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title(name)
  plt.grid(False)

  plt.savefig(opt.folder + name+".png")
  plt.show(block = False)




def plotOverlay2(values,fignum,name,opt):
  img = cv2.imread(opt.folder+'images/image0.jpg')

  fig, ax = plt.subplots(1, 1)
  ax.imshow(img)
  ax.scatter(values[1],values[0], color="r", s=10, alpha=0.15)

  plt.xlabel('x')
  plt.ylabel('y')
  plt.title(name)
  plt.grid(False)

  plt.savefig(opt.folder + name+".png")
  plt.show(block = False)

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap


if __name__ == "__main__":
  main()