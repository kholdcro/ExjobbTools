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
  (options, args) = parser.parse_args()

  orbPos, keyTime = getORBData(options)
  mocPos, mocRot, mocTime = getMocapData(options)
  orbTime = getOrbTimes(options)

  # plot3D(mocPos,2,"MocapPath")
  
  mocPos = pruneMocData(mocPos,0.05)
  mocPos = pruneMocData(mocPos,0.02)

  mCap = getMocCaps(mocPos, 0.001)
  mocPos = smoothData(mocPos, 3)


  oCap = [202, 1315]    #for "BigFlight"

  mCap[1] = mCap[1]-300
  offset = mocTime[mCap[0]]-orbTime[oCap[0]]

  mocPos  = mocPos[:,mCap[0]:mCap[1]]
  mocTime = mocTime[mCap[0]:mCap[1]]
  mocRot  = mocRot[:,mCap[0]:mCap[1]]

  orbPos, mocRange = interpOrbData(mocPos, orbPos,
                   mocTime, keyTime, offset)

  mocPos  = mocPos[:,mocRange[0]:mocRange[1]]
  mocTime = mocTime[mocRange[0]:mocRange[1]]
  mocRot  = mocRot[:,mocRange[0]:mocRange[1]]  

  # writeSmoothData(options, mocPos, mocRot, mocTime)

  # oflightTime = orbTime[oCap[1]]-orbTime[oCap[0]]
  oflightTime = keyTime[-1] - keyTime[0]

  oFrames = orbPos.shape[1]
  mFrames = mCap[1]-mCap[0]
  mflightTime = mocTime[-1]-mocTime[0]

  print("ORB:\t" , "Frames: ", oFrames, "\tFlightTime:", oflightTime, "\tFPS: ", oFrames/oflightTime)
  print("MOCAP:\t","Frames: ", mFrames, "\tFlightTime:", mflightTime, "\tFPS: ", mFrames/mflightTime)

  # printSample()

  print(np.array([[1,1,0],[0.5,0.5,0]]).shape)
  print(orbPos.T.shape)
  print(mocPos.T.shape)

  print("Calculating...")
  R, t, x = calcMatrices(orbPos.T, mocPos.T)
  print("done!")

  newOrb = np.dot(R,orbPos).T+t 

  getErrors(x, newOrb, mocPos.T)
  writeToFile(options, x, newOrb, mocPos.T)

  # plot3D(orbPos,1,"OrbPath")
  # plot3D(mocPos,3,"SmoothedMocap")
  doublePlot([newOrb[:,0].T,newOrb[:,1].T],"Orb", [mocPos[0],mocPos[1]],
              "Mocap",3,"x vs y", options, xlabel='x', ylabel='y')
  doublePlot([newOrb[:,0].T,newOrb[:,2].T],"Orb", [mocPos[0],mocPos[2]],
              "Mocap",4,"x vs z", options, xlabel='x', ylabel='z')
  doublePlot([newOrb[:,1].T,newOrb[:,2].T],"Orb", [mocPos[1],mocPos[2]],
              "Mocap",5,"y vs z", options, xlabel='y', ylabel='z')  
  doublePlot3D(newOrb.T,"Orb",mocPos,"Mocap",6,"Double_Plot", options)
  # doublePlot3D(orbPos,"Orb",mocPos,"Mocap",5,"Double_Plots", options)

  input("Press [enter] to continue.")
  print("\nTotal Time:\t" + str(round(time.time()-start,3)))


def getORBData(opt):
  with open(opt.folder+'analyze/KeyFrameTrajectory.txt', 'r') as f:
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

  print(position.shape)
  return position, time


def getOrbTimes(opt):
  with open(opt.folder+'times.txt', 'r') as f:
    times = []
    for line in f:
        times.append(line)
  times = [float(i) for i in times]
  times = np.array(times)
  return times


def getMocapData(opt):
  with open(opt.folder+'analyze/groundTruth.csv', 'r') as f:
    reader = csv.reader(f, delimiter=",")
    moData = list(reader)
  nMoData = np.array(moData)

  frame       = [int(i) for i in nMoData[:,0]]
  time        = [float(i) for i in nMoData[:,1]]
  rot_x       = [float(i) for i in nMoData[:,2]]
  rot_y       = [float(i) for i in nMoData[:,3]]
  rot_z       = [float(i) for i in nMoData[:,4]]
  rot_w       = [float(i) for i in nMoData[:,5]]
  pos_x       = [float(i) for i in nMoData[:,6]]
  pos_y       = [float(i) for i in nMoData[:,7]]
  pos_z       = [-float(i) for i in nMoData[:,8]]
  markerError = [float(i) for i in nMoData[:,9]]

  position = np.array([pos_x, pos_y, pos_z])
  rotation = np.array([rot_x, rot_y, rot_z, rot_w])

  return position, rotation, time

def writeSmoothData(options, mocPos, mocRot, mocTime):
  print(mocPos.shape)
  print(mocRot.shape)
  print(np.array(mocTime).shape)
  print(np.array([mocTime]).shape)
  data = np.concatenate((np.array([mocTime]),mocPos), axis=0)
  data = np.concatenate((data,mocRot), axis=0)
  print(data.shape)
  print(data[0])
  print(len(data[1]))
  with open('newMocap.csv', 'w') as f:
    np.savetxt(f, data.T, fmt='%.5f')


def pruneMocData (position, threshold):
  prevDist = np.sqrt(position[0,0]**2+position[1,0]**2+position[2,0]**2)
  for i,pos in enumerate(position.T):
    dist = np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)
    if abs(dist-prevDist) > threshold:
      x_interp = position[0,i-1]+(position[0,i+1]-position[0,i-1])/2
      position[:,i] = position[:,i-1]+(position[:,i+1]-position[:,i-1])/2
    dist = np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)
    prevDist = dist
  return position


def smoothData(position, N):
  for i, pos in enumerate(position.T):
    if(i<2 or i>= len(position.T)-2):
      continue
    # print(position[:,i-1])
    if N ==3:
      position[:,i] = (position[:,i-1] + pos + position[:,i+1])/3
    else:
      position[:,i] = (position[:,i-2]+position[:,i-1] + pos + position[:,i+1]+ position[:,i+2])/5
  # position[:,0] = np.convolve(position[:,0], np.ones((N,))/N, mode='valid')
  return position


def getMocCaps(position, threshold):
  zn = position[2,0]
  for i, z in enumerate(position[2,:]):
    if(abs(z)-abs(zn) > threshold):
      moStart = i
      break
  
  zn = position[2,-1]
  for i, z in enumerate(reversed(position[2,:])):
    if(abs(z)-abs(zn) > threshold):
      moLast = len(position[2,:])-i
      break

  return [moStart, moLast]


def getDistance(position):
  for i,pos in enumerate(position.T):
    dist = np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)
    totDist += dist
  avgDist = totDist/position.shape[1]
  return totDist


def matchFPS(mocPos, orbPos, mocTime, orbTime):
  t = 0
  print(len(mocTime))
  print(len(orbTime))
  print("derp")
  print(mocPos.shape)
  print(orbPos.shape)
  newPos = []
  prevOT = -1.
  for i, oT in enumerate(orbTime):
    botM = 0
    for j, mT in enumerate(mocTime):
      if(mT>oT) and (j > 0):
        botM = mocTime[j-1]
        ratio = (oT-botM)/(mT-botM)
        # print((oT-botM)/(mT-botM), oT, mT, botM)
        tmpPos = orbPos[:,i]+ (mocPos[:,j]-mocPos[:,j-1])*ratio
        newPos.append(tmpPos)
        t+=1
        break
  print("Faak", t)
  print(len(newPos))
  
  print("orb:", t)
  return mocPos

def interpOrbData(mocPos, orbPos, mocTime, keyTime, offset):
  t = 0
  newPos = []
  mocTime[:] = np.array(mocTime) - offset
  print("MOC 0: ", mocTime[0])
  keyTime[:] = np.array(keyTime)  
  i=0
  frogger = True
  start = 0
  for j, mT in enumerate(mocTime):
    if j==0 or mT < keyTime[0]:
      continue
    elif frogger:
      start = j
      frogger = False
    if (mT > keyTime[i+1]):
      i+=1
    if i >= orbPos.shape[1]:
      print("ORBSHAPE: ", orbPos.shape[1])
      break
    ratio = (mT-mocTime[j-1])/(keyTime[i+1]-keyTime[i])
    pos = orbPos[:,i]+ (orbPos[:,i+1]-orbPos[:,i])*ratio
    newPos.append(pos)
  newPos = np.array(newPos)
  return newPos.T, [start-1,j]


def leastSquareError(x, oPos, mPos):
  R = np.array([[x[0],x[1],x[2]],[x[4],x[5],x[6]],[x[7],x[8],x[9]]])
  t = np.array([[x[10],x[11],x[12]]])
  t = np.repeat(t.T, oPos.shape[0], axis=1)
  error = np.linalg.norm((np.dot(R,oPos.T)+t)- mPos.T)
  return error


def calcMatrices(oPos, mPos):
  result = minimize(
      fun=leastSquareError,
      x0=np.ones(13),
      args=(oPos, mPos),
      method='CG',
      # method='BFGS',
      options={
          'maxiter': 50
      })
  print("Success: ", result.success)
  print("Message: ", result.message)
  print("Iterats: ", result.nit)
  x = result.x

  R = np.array([[x[0],x[1],x[2]],[x[4],x[5],x[6]],[x[7],x[8],x[9]]])
  t = np.array([x[10],x[11],x[12]])

  print(R, t)
  return R, t, x

def rmse(predictions, targets):
  return np.sqrt(((predictions - targets) ** 2).mean())

def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def totalError(predictions, targets):
    return (np.abs(predictions - targets)).sum()

def getLength(pos):
  distance = 0
  for i in range(1, pos.shape[0]):
    distance += np.sqrt((pos[i,0]-pos[i-1,0])**2 +
                 (pos[i,1]-pos[i-1,1])**2 +
                 (pos[i,2]-pos[i-1,2])**2)
  return distance

def getErrors(x, oPos, mPos):
  lse = leastSquareError(x, oPos, mPos)
  print("Total:\t\t", totalError(oPos, mPos))
  print("Euclid:\t\t", lse)
  print("Euclid/n:\t", lse/oPos.shape[0])
  print("RMSE:\t\t", rmse(oPos, mPos))
  print("MSE:\t\t", mse(oPos, mPos))
  print("Length: ", getLength(mPos))

def printSample():
  sample = np.array([[1,1,0],[0.5,0.5,0]])
  sample2 = np.array([[2,2,0],[1,1,0]])
  R, t = calcMatrices(sample, sample2)

  print(np.dot(R,sample[0])+t)
  print(np.dot(R,sample[1])+t)

def plotThings(pos,figNum,name):
  fig = plt.figure(figNum)
  plt.plot(pos[0],pos[1])
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Path')
  plt.grid(True)
  print(pos[0,0], pos[1,0])
  circle1 = plt.Circle((pos[0,0], pos[2,0]), 0.1, color='g')
  circle2 = plt.Circle((pos[0,-1], pos[2,-1]), 0.1, color='r')

  fig = plt.gcf()
  # ax = fig.gca()

  ax.add_artist(circle1)
  ax.add_artist(circle2)
  plt.savefig(name+".png")
  plt.show()


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