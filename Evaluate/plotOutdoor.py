import sys, time
import numpy as np
# import cv2
# assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import glob
from optparse import OptionParser
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import scipy.fftpack
from scipy.optimize import minimize, least_squares

def main():
  start = time.time()
  parser = OptionParser()  
  parser.add_option("-f", "--folder", dest="folder",
                  help="Choose I/O Folder", metavar="FILE")
  parser.add_option("-n", "--name", dest="name",
                  help="Name of Plots", default="Plot")
  (options, args) = parser.parse_args()

  orbPos, keyTime = getORBData(options)
  orbTime = getOrbTimes(options)

  fignum = 0
  plotThings(orbPos,fignum, options.name, options)
  fignum += 1
  plot3D(orbPos,fignum, options.name+"3D")


  input("Press [enter] to continue.")
  print("\nTotal Time:\t" + str(round(time.time()-start,3)))

def getORBData(opt):
  with open(opt.folder+'/KeyFrameTrajectory.txt', 'r') as f:
    reader = csv.reader(f, delimiter=" ")
    orbData = list(reader)

  norbData = np.array(orbData)
  time       = [float(i) for i in norbData[:,0]]
  pos_x      = [float(i) for i in norbData[:,1]]
  pos_y      = [-float(i) for i in norbData[:,2]]
  pos_z      = [float(i) for i in norbData[:,3]]
  rot_x      = [float(i) for i in norbData[:,4]]
  rot_y      = [float(i) for i in norbData[:,5]]
  rot_z      = [float(i) for i in norbData[:,6]]  
  rot_w      = [float(i) for i in norbData[:,7]]  

  position = np.array([pos_x, pos_y, pos_z])

  return position, time


def getOrbTimes(opt):
  with open(opt.folder+'times.txt', 'r') as f:
    times = []
    for line in f:
        times.append(line)
  times = [float(i) for i in times]
  times = np.array(times)
  return times


def getDistance(position):
  for i,pos in enumerate(position.T):
    dist = np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)
    totDist += dist
  avgDist = totDist/position.shape[1]
  return totDist


def getLength(pos):
  distance = 0
  for i in range(1, pos.shape[0]):
    distance += np.sqrt((pos[i,0]-pos[i-1,0])**2 +
                 (pos[i,1]-pos[i-1,1])**2 +
                 (pos[i,2]-pos[i-1,2])**2)
  return distance


def plotThings(pos,figNum,name,opt):
  fig = plt.figure(figNum)
  plt.plot(pos[0],pos[2])
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Path')
  plt.grid(True)
  print(pos.shape)
  print(pos[0,0], pos[1,0])
  circle1 = plt.Circle((pos[0,0], pos[2,0]), 0.1, color='g')
  circle2 = plt.Circle((pos[0,-1], pos[2,-1]), 0.1, color='r')

  fig = plt.gcf()
  ax = fig.gca()

  ax.add_artist(circle1)
  ax.add_artist(circle2)
  plt.savefig(opt.folder + name+".png")
  plt.show(block = False)


def plot3D(pos,figNum,name):
  fig = plt.figure(name)
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(pos[0],pos[1], pos[2])
  plt.xlabel('x')
  plt.ylabel('y')
  # plt.zlabel('z')
  plt.title(name)
  plt.grid(True)
  ax.scatter(pos[0,0], pos[1,0], pos[2,0], color="g", s=100)
  ax.scatter(pos[0,-1], pos[1,-1], pos[2,-1], color="r", s=100)
  plt.show(block = False)  


def doublePlot(pos1,name1,pos2,name2,figNum,name, options, xlabel='x', ylabel='y'):
  fig = plt.figure(figNum)
  ax = fig.gca()
  line1 = ax.plot(pos1[0],pos1[1], color='c', label=name1)
  line2 = ax.plot(pos2[0],pos2[1], color='black', label=name2)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(name)
  plt.grid(True)
  plt.savefig(options.folder+"/analyze/"+name+".png")
  plt.show(block = False)  


def doublePlot3D(pos,name1,pos2,name2,figNum,name, options):
  fig = plt.figure(name)
  ax = fig.add_subplot(111, projection='3d')
  line1 = ax.plot(pos[0],pos[1], pos[2], color='c', label=name1)
  line2 = ax.plot(pos2[0],pos2[1], pos2[2], color='black', label=name2)
  plt.xlabel('x')
  plt.ylabel('y')
  # plt.zlabel('z')
  plt.title(name)
  plt.grid(True)
  ax.scatter(pos[0,0], pos[1,0], pos[2,0], color="g", s=100)
  ax.scatter(pos[0,-1], pos[1,-1], pos[2,-1], color="r", s=100)
  ax.legend()

  for i in range(pos.shape[1]):
    if np.mod(i,50):
      continue
    ax.scatter(pos[0,i], pos[1,i], pos[2,i], color="c", s=1)
    ax.scatter(pos2[0,i], pos2[1,i], pos2[2,i], color="black", s=1)
    ax.plot([pos[0,i],pos2[0,i]], 
            [pos[1,i],pos2[1,i]], 
            [pos[2,i],pos2[2,i]], color="r")
  plt.savefig(options.folder+"/analyze/"+name+".png")
  plt.show(block = False)  


def writeToFile(options, x,oPos, mPos):
  lse = leastSquareError(x, oPos, mPos)
  with open(options.folder+"/analyze/errorCalculations.txt", "w") as f: 
    f.write("Total:\t\t{0:.3f}\n".format(totalError(oPos, mPos)))
    f.write("Euclid:\t\t{0:.3f}\n".format(lse))
    f.write("Euclid_avg:\t{0:.3f}\n".format(lse/oPos.shape[0]))
    f.write("RMSE:\t\t{0:.4f}\n".format(rmse(oPos, mPos)))
    f.write("MSE:\t\t{0:.5f}\n".format(mse(oPos, mPos)))
    f.write("Length:\t\t{0:.3f}".format(getLength(mPos)))


if __name__ == "__main__":
  main()