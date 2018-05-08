import sys, time
import numpy as np
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import glob
from optparse import OptionParser
import copy


def main():
  start = time.time()

  parser = OptionParser()  
  parser.add_option("-f", "--folder", dest="folder",
                  help="Choose I/O Folder", metavar="FILE")
  (options, args) = parser.parse_args()

  cbDims = (6,7)

  mtx = np.zeros((3,3))
  mtx[0,0] = 301.887922
  mtx[1,1] = 301.720198
  mtx[0,2] = 481.204306
  mtx[1,2] = 478.820797
  mtx[2,2] = 1.

  dist = np.array([-0.017766, 0.000774, 0.007102, -0.003659], dtype=np.float32)

  print("Camera Matrix:")
  print(mtx)
  print("\nDistortion Parameters:")
  print(dist)

  # testImage(options, mtx, dist)
  # print("Breaking Early")
  # return

  dispExample(options, mtx, dist, "image", cbDims)

  cv2.destroyAllWindows()
  print("\nTotal Time:\t" + str(round(time.time()-start,3)))



def dispExample(opt, mtx, dist, name, cbDims):
  # img = cv2.imread(opt.folder+'/fishTest.jpg', cv2.IMREAD_COLOR)
  img = cv2.imread(opt.folder+'/test.jpg', cv2.IMREAD_COLOR)
  h,  w = img.shape[:2]
  half_w = int(w/2)
  
  xx = np.linspace(-half_w+1,half_w,num=w)
  yy = np.linspace(-half_w+1,half_w,num=w)

  np.set_printoptions(precision=2)  
  # printExamples(mtx, dist, half_w)

  # testThresholds(mtx, dist, w, h, opt)
  # return

  xMap = np.zeros((w,h), np.int32)
  yMap = np.zeros((w,h), np.int32)

  xMap = np.zeros((w,h))
  yMap = np.zeros((w,h))

  r = np.zeros((w,h))

  threshold = np.pi/2-0.2
  threshold = np.pi/2-0.15
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      x,y, r[i,j] = distortPoints(i, j, dist, mtx, thresh=threshold, old=False)
      x,y = rectifyPoint(x,y,mtx)
      xMap[i,j] = x
      yMap[i,j] = y
    if(np.mod(i,100) ==0):
      print("Iteration: {}".format(i))

  print("\nNumber of Elements: {}".format(np.count_nonzero(xMap)))
  print("xMap: \n",xMap)

  cv2.imwrite(opt.folder+'/xMap0.jpg', ((xMap)/(np.max(xMap))+0.5)*127.)
  cv2.waitKey(500)

  writeToFile(xMap, "xMapInt", opt)

  xMap = np.round(xMap)
  yMap = np.round(yMap)

  xMap = xMap.astype(int)
  yMap = yMap.astype(int)

  minX = np.min(xMap)
  maxX = np.max(xMap)
  minY = np.min(yMap)
  maxY = np.max(yMap)
  print("Max Scale: {}".format(np.max(r)))
  print("x Limit: \t [{}, {}]".format(minX, maxX))
  print("y Limit: \t [{}, {}]".format(minY, maxY))

  cv2.imwrite(opt.folder+'/xMap1.jpg', ((xMap)/(maxX))*255.)
  cv2.waitKey(500)

  mapScale = np.min([minX, minY])

  xMap[xMap!=0] -= mapScale
  yMap[yMap!=0] -= mapScale

  minX = np.min(xMap)
  maxX = np.max(xMap)
  minY = np.min(yMap)
  maxY = np.max(yMap)
  print("After Scale: {}".format(np.max(mapScale)))
  print("x Limit: \t [{}, {}]".format(minX, maxX))
  print("y Limit: \t [{}, {}]".format(minY, maxY))  

  writeToFile(xMap, "xMapScale", opt)

  cv2.imwrite(opt.folder+'/xMap2.jpg', ((xMap)/(maxX))*255.)
  cv2.waitKey(500)

  img2 = np.zeros((int(maxX-mapScale+1), int(maxY-mapScale+1),3), np.uint8)
  img3 = np.zeros((int(maxX-mapScale+1), int(maxY-mapScale+1),2), np.uint8)

  img2 = np.zeros((int(maxX+1), int(maxY+1),3), np.uint8)
  img3 = np.zeros((int(maxX+1), int(maxY+1),2), np.uint8)  


  print("Image Shape:  {}".format(img.shape))
  print("Image2 Shape: {}".format(img2.shape))
  
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if(xMap[i,j]) and yMap[i,j]:
        img2[xMap[i,j],yMap[i,j]] = img[i,j]
        img3[xMap[i,j],yMap[i,j]] = [i,j]


  # writeToFile(img3[:,:,0], "mapped thing", opt)

  print("Done")
  cv2.imwrite(opt.folder+'/manual.jpg',img2)
  cv2.waitKey(500)
  # derp = (xMap/maxX)*255
  # print(derp)
  # cv2.imwrite(opt.folder+'/xMap.jpg', derp )
  cv2.waitKey(500)
  print("Done")

  # cv2.imwrite(opt.folder+'/dottest.jpg',img)
  # cv2.waitKey(50)

def undistortPoints(x, y, dist, K):
  if x == 0 and y ==0:
    return 0,0,0

  r = np.sqrt(x**2 + y**2)
  # if(r>half_w):
  #   return 0,0,0
  theta = np.arctan(r)
  # theta = np.arctan2(y,x)
  theta_d = theta*(1+dist[0]*theta**2+dist[1]*theta**4
                +dist[2]*theta**6+dist[3]*theta**8)

  # if(r ==0):
  #   print(x,y,r,theta, theta_d)
  xp = (theta_d/r)*x
  yp = (theta_d/r)*y
  # print(xp, yp)
  u = K[0,0]*xp +K[0,2]
  v = K[1,1]*yp +K[1,2]
  return u,v, r


def distortPoints(x,y, dist, K, thresh=(np.pi/2-0.2), old=False):
  xp = (x - K[0,2])/K[0,0]  
  yp = (y - K[1,2])/K[1,1]

  theta_d = np.sqrt(xp**2 + yp**2)
  # print(theta_d)

  if(np.abs(theta_d) > thresh):
    # print("Exiting", np.pi/2)
    return 0.,0.,0.
    # return 0.,0.,-1.
  
  scale = 1.
  theta = copy.deepcopy(theta_d)
  if (theta_d > .00000001):
    for j in range(100):
      # print(theta)
      theta2 = theta*theta
      theta4 = theta2*theta2
      theta6 = theta2*theta4
      theta8 = theta4*theta4

      k0_theta2 = dist[0]*theta2
      k1_theta4 = dist[1]*theta4
      k2_theta6 = dist[2]*theta6
      k3_theta8 = dist[3]*theta8

      if not old:
        theta_fix = ((theta* (1. + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d)/
                            (1. + 3.*k0_theta2 + 5.*k1_theta4 + 7.*k2_theta6 + 9.*k3_theta8))

        theta = theta - theta_fix
        if (np.abs(theta_fix) < .00000001):
          break

      elif old:
        theta = theta_d / (1.+k0_theta2+k1_theta4+k2_theta6+k3_theta8)

    # if(np.abs(theta) > thresh):
    #   # print("Exiting2", theta)
    #   return 0.,0.,-1.
    scale = np.tan(theta) / theta_d

  if np.abs(scale) > 10000:
    print("x: {} \t, y: {} \t, Scale: {}".format(x, y, scale))

  u = xp * scale
  v = yp * scale

  # if u < 0. or u > 960.:
  #   return 0.,0.,0.

  # if v < 0. or v > 960.:
  #   return 0.,0.,0.
  return u,v, scale


def rectifyPoint(x, y, mtx):
  # if x ==0 and y ==0:
  #   return 0., 0.
  x = x*mtx[0,0] + mtx[0,2]
  y = y*mtx[1,1] + mtx[1,2]
  return x, y


def printExamples(mtx, dist, half_w):
  xt,yt, rt = undistortPoints(10., 10., dist, mtx)
  print("Ours:\t [{}, {}],\t [{:.2f} {:.2f}],\t r: {}".format(10., 10., xt, yt, rt))
  print(xt)

  point = np.zeros((1,1,2), dtype=np.float32)
  point[0] = [10, 10]
  unpoint = cv2.fisheye.distortPoints(point, mtx, np.array([dist]))  
  print("Theirs:\t {},\t {}".format(point[0,0], unpoint[0,0]))

  print("UndistortPoints\n")
  point = np.zeros((3,1,2), dtype=np.float32)
  point[0] = [10, 10]
  point[1] = [100, 100]
  point[2] = [-200, -200]
  unpoint = cv2.fisheye.undistortPoints(point, mtx, dist)
  print(unpoint.shape)
  print("Theirs:\t [{}, {}],\t\t [{:.2f}, {:.2f}]".format(point[0,0,0],point[0,0,1], unpoint[0,0,0], unpoint[0,0,1]))
  xt,yt, rt = distortPoints(10., 10., dist, mtx)
  print("Ours:\t [{}, {}],\t\t [{:.2f}, {:.2f}],\t r: {:.2f}".format(10., 10., xt, yt, rt))

  print("Theirs:\t [{}, {}],\t [{:.2f}, {:.2f}]".format(point[1,0,0],point[1,0,1], unpoint[1,0,0], unpoint[1,0,1]))
  xt,yt, rt = distortPoints(100., 100., dist, mtx)
  print("Ours:\t [{}, {}],\t [{:.2f}, {:.2f}],\t r: {:.2f}".format(100., 100., xt, yt, rt))

  print("Theirs:\t [{}, {}],\t [{:.2f}, {:.2f}]".format(point[2,0,0],point[2,0,1], unpoint[2,0,0], unpoint[2,0,1]))
  xt,yt, rt = distortPoints(-200., -200., dist, mtx)
  print("Ours:\t [{}, {}],\t [{:.2f}, {:.2f}],\t r: {:.2f}".format(-200., -200., xt, yt, rt))  
  print(type(xt), type(yt)) 


def testThresholds(mtx, dist, w,h, options):

  numData = 10
  threshold = np.linspace(0.1, np.pi/2, num=numData)
  threshold = np.linspace(1.43, 1.49, num=numData)
  # threshold = np.logspace(1,0,num=numData, base=(np.pi/2))
  # print(np.logspace(1,0,num=numData, base=(np.pi/2)))
  xScale = np.zeros((numData,2))
  yScale = np.zeros((numData,2))
  absScale = np.zeros(numData)

  print(xScale)

  for ii, thresh in enumerate(threshold):
    print("\nRun {},\tThreshold: {}".format(ii, thresh))
    xMap = np.zeros((w,h))
    yMap = np.zeros((w,h))
    r = np.zeros((w,h))
    
    for i in range(w):
      for j in range(h):
        x,y, r[i,j] = distortPoints(i, j, dist, mtx, thresh=thresh)
        x,y = rectifyPoint(x,y,mtx)
        xMap[i,j] = x
        yMap[i,j] = y
      if(np.mod(i,100) ==0):
        print("Iteration: {}".format(i))

    print("\nNumber of Elements: {}".format(np.count_nonzero(xMap)))

    xMap = np.round(xMap)
    yMap = np.round(yMap)

    xMap = xMap.astype(int)
    yMap = yMap.astype(int)

    minX = np.min(xMap[xMap!=0])
    maxX = np.max(xMap)
    minY = np.min(yMap[yMap!=0])
    maxY = np.max(yMap)
    print("Max Scale: {}".format(np.max(r)))
    print("x Limit: \t [{}, {}]".format(minX, maxX))
    print("y Limit: \t [{}, {}]\n".format(minY, maxY))

    xScale[ii] = [minX, maxX]
    yScale[ii] = [minY, maxY]
    absScale[ii] = np.max(r)

    print(xScale)

  with open(options.folder+"/thresholdInfo.txt", "w") as f:
    f.write("Threshold:\n")
    np.savetxt(f, threshold, fmt='%.5f')
    f.write("xScale:\n")
    np.savetxt(f, xScale, fmt='%.2f')
    f.write("yScale:\n")
    np.savetxt(f, yScale, fmt='%.2f')
    f.write("absoluteScale:\n")
    np.savetxt(f, absScale, fmt='%.2f')    

  return


def testImage(opt, mtx, dist):

  img = cv2.imread(opt.folder+'/undistortTest/test.jpg', cv2.IMREAD_COLOR)
  h,  w = img.shape[:2]
  half_w = int(w/2)

  balance=1.
  dim1 = img.shape[:2]  #dim1 is the dimension of input image to un-distort
  dim2=None
  dim3=None

  dim2 = tuple(np.multiply(1,dim1))
  dim3 = tuple(np.multiply(1,dim1))


  m_K = mtx.copy()
  new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(m_K, dist, dim2, np.eye(3), balance=balance)
  # new_K = mtx.copy()
  # new_K[0,2] = xDim/2
  # new_K[1,2] = yDim/2
  map1, map2 = cv2.fisheye.initUndistortRectifyMap(m_K, dist, np.eye(3), new_K, dim3, cv2.CV_16SC2)
  undistorted_img2 = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

  edge = w-1.
  diff = w-edge
  print(diff)

  point = np.zeros((10,1,2), dtype=np.float32)
  point[0] = [w/4, h-h/4]
  point[1] = [w/2, diff]
  point[2] = [diff, h/2]
  point[3] = [edge, h/2]
  point[3] = [322.+w/2,322.+h/2]
  point[4] = [w/2, edge]
  point[5] = [w/4, h/4]
  point[6] = [w-w/3, h-h/2.5]
  point[7] = [3/5*w, 3/4*h]
  point[8] = [3/7*w, 2/5*h]
  point[9] = [3/7*w, 3/5*h]

  cvFish = cv2.fisheye.undistortPoints(point,m_K,dist, np.eye(3),m_K)
  print("lenPoint: {}".format(len(point)))
  print("size Point: {}    {}".format(point.shape[0],point.shape[2]))
  # goodFish = np.zeros((point.shape))
  goodFish = np.zeros((point.shape[0],point.shape[2]))
  for i in range(len(point)):
    x, y, z = distortPoints(point[i,0,0],point[i,0,1],dist, m_K, thresh=np.pi/2, old=False)
    if(z!=-1):
      x, y = rectifyPoint(x, y, m_K)
      if(checkInImage(x,y,img.shape)):
        goodFish[i] = [x,y]
      else:
        goodFish[i,:] = None
    else:
      goodFish[i,:] = None

  oldFish = np.zeros((point.shape[0],point.shape[2]))
  for i in range(len(point)):
    x, y, z = distortPoints(point[i,0,0],point[i,0,1],dist, m_K, thresh=np.pi/2, old=True)
    if(z!=-1):
      x, y = rectifyPoint(x, y, m_K)
      if(checkInImage(x,y,img.shape)):
        oldFish[i] = [x,y]
      else:
        oldFish[i,:] = None
    else:
      oldFish[i,:] = None      

  print("Point, \t Undistorted")
  for i in range(len(point)):
    print("[{}, {}] \t [{}, {}]".format(point[i,0,0],point[i,0,1],cvFish[i,0,0],cvFish[i,0,1]))  

  # map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, R=np.eye(3), P=mtx , size=(w,h), m1type=cv2.CV_16SC2)
  map1, map2 = cv2.fisheye.initUndistortRectifyMap(m_K, dist, np.eye(3), m_K, dim3, cv2.CV_16SC2)
  cvFish_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
  goodFish_img = cvFish_img.copy()
  oldFish_img = cvFish_img.copy()

  colours = [(0,0,255),(255,0,255),(255,0,0),
             (255,255,0),(0,255,0),(255,255,255),
             (0,255,255),(0,0,0),(0,127,255)]

  for i, colour in enumerate(colours):
    cv2.circle(img,(int(point[i,0,0]),int(point[i,0,1])), 20, colour, 4)
    cv2.circle(cvFish_img,(int(cvFish[i,0,0]),int(cvFish[i,0,1])), 20, colour, 4)
    if not np.isnan(goodFish[i,0]):
      cv2.circle(goodFish_img,(int(goodFish[i,0]),int(goodFish[i,1])), 20, colour, 4)
    if not np.isnan(oldFish[i,0]):
      cv2.circle(oldFish_img,(int(oldFish[i,0]),int(oldFish[i,1])), 20, colour, 4)      

  cv2.imwrite(opt.folder+'/undistortTest/dottest.jpg',img)
  cv2.waitKey(50)
  cv2.imwrite(opt.folder+'/undistortTest/cvFish.jpg',cvFish_img)
  cv2.waitKey(50)
  cv2.imwrite(opt.folder+'/undistortTest/goodFish.jpg',goodFish_img)
  cv2.waitKey(50)
  cv2.imwrite(opt.folder+'/undistortTest/oldFish.jpg',oldFish_img)
  cv2.waitKey(1)

def checkInImage(x,y, shape):
  # print("derp", x,y,shape)
  if x < 0. or x > shape[0]:
    return False
  if y < 0. or y > shape[1]:
    return False  
  return True

def writeToFile(variable, name, options):
  with open(options.folder+"/"+name+".txt", "w") as f:
    np.savetxt(f, variable, fmt='%.2f')


if __name__ == "__main__":
  main() 