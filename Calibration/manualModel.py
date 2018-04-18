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


  dispExample(options, mtx, dist, "image", cbDims)

  cv2.destroyAllWindows()
  print("\nTotal Time:\t" + str(round(time.time()-start,3)))



def dispExample(opt, mtx, dist, name, cbDims):
  # img = cv2.imread(opt.folder+'/fishTest.jpg', cv2.IMREAD_COLOR)
  img = cv2.imread(opt.folder+'/test.jpg', cv2.IMREAD_COLOR)
  h,  w = img.shape[:2]
  half_w = int(w/2)
  # half_w = 906./2.

  # xx = np.linspace(-half_w+1,half_w,num=w)
  # yy = np.linspace(-half_w+1,half_w,num=w)
  xx = np.linspace(-half_w+1,half_w,num=w)
  yy = np.linspace(-half_w+1,half_w,num=w)
  # xx = np.linspace(-w+1,w,num=w)
  # yy = np.linspace(-w+1,w,num=w)

  print(np.min(xx))
  print(np.min(yy))

  # xx = np.linspace(-np.pi/2,np.pi/2,num=w)
  # yy = np.linspace(-np.pi/2,np.pi/2,num=w)

  np.set_printoptions(precision=2)
  
  printExamples(mtx, dist, half_w)

  # return

  xMap = np.zeros((w,h), np.int32)
  yMap = np.zeros((w,h), np.int32)

  xMap = np.zeros((w,h))
  yMap = np.zeros((w,h))

  r = np.zeros((w,h))

  # cv2.imshow("Gaaaakkk", img)
  # cv2.waitKey(0)

  # for i in range(img.shape[0]):
  #   for j in range(img.shape[1]):
  #     x,y, r[i,j] = distortPoints(i, j, dist, mtx, half_w)
  #     # x, y = i, j
  #     # print(x,xn, " : ", y, yn)
  #     xMap[i,j] = int(round(x))
  #     yMap[i,j] = int(round(y))
  #   if(np.mod(i,100) ==0):
  #     print("Iteration: {}".format(i))

  if(False):
    point = np.zeros((img.shape[0]**2,1,2), dtype=np.float32)
    ii=0
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        point[ii,0] = [i-half_w,j-half_w]
        ii+=1
    print(point)
    unpoint = cv2.fisheye.undistortPoints(point, mtx, dist)
  
    minX = np.min(unpoint[:,0,0])
    maxX = np.max(unpoint[:,0,0])
    minY = np.min(unpoint[:,0,1])
    maxY = np.max(unpoint[:,0,1])
    print(minX, maxX)
    print(minY, maxY)

    # img2 = np.zeros((int(maxX-minX+1), int(maxY-minY+1),3), np.uint8)
    img2 = np.zeros(img.shape, np.uint8)
    ii=0
    for i in range(img.shape[0]):
      unpoint[ii,0,0] = (unpoint[ii,0,0] -mtx[0,2])/mtx[0,0]
      unpoint[ii,0,1] = (unpoint[ii,0,1]-mtx[1,2])/mtx[1,1]
      for j in range(img.shape[1]):
        if(unpoint[ii,0,0] <960) and (unpoint[ii,0,1]<960):
          if(unpoint[ii,0,0] >0) and (unpoint[ii,0,1] >0):
            img2[unpoint[ii,0,0],unpoint[ii,0,1]] = img[i,j]
            ii += 1
    print(ii)

    cv2.imwrite(opt.folder+'/manual.jpg',img2)
    cv2.waitKey(500)
    return
  else:
    # for i, xn in enumerate(xx):
    #   for j, yn in enumerate(yy):
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        # x,y, r[i,j] = distortPoints(xn, yn, dist, mtx, half_w)
        x,y, r[i,j] = distortPoints(i, j, dist, mtx, half_w)
        x,y = rectifyPoint(x,y,mtx)
        # xMap[i,j] = int(round(x))
        # yMap[i,j] = int(round(y))
        xMap[i,j] = x
        yMap[i,j] = y
        # xMap[i,j] = x
        # yMap[i,j] = y
      if(np.mod(i,100) ==0):
        print("Iteration: {}".format(i))

  # xMap = xMap/10
  # yMap = yMap/10

  print("Number of Elements: {}".format(np.count_nonzero(xMap)))
  print("xMap: \n",xMap)

  cv2.imwrite(opt.folder+'/xMap0.jpg', ((xMap)/(np.max(xMap))+0.5)*127.)
  cv2.waitKey(500)

  writeToFile(xMap, "xMapInt", opt)
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
  xMap -= mapScale
  yMap -= mapScale

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


  print("Image Shape:  {}".format(img.shape))
  print("Image2 Shape: {}".format(img2.shape))
  
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if(xMap[i,j]) and yMap[i,j]:
        img2[xMap[i,j],yMap[i,j]] = img[i,j]
        img3[xMap[i,j],yMap[i,j]] = [i,j]


  writeToFile(img3[:,:,0], "mapped thing", opt)

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

def undistortPoints(x, y, dist, K, half_w):
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


def distortPoints(x,y, dist, K, half_w):
  # x = float(x)
  # y = float(y)
  xp = (x - K[0,2])
  yp = (y - K[1,2])

  # if np.sqrt(xp**2 + yp**2) >800.:
  #   return 0.,0.,0.

  xp /= K[0,0]  
  yp /= K[1,1]

  theta_d = np.sqrt(xp**2 + yp**2)
  # print(xp, yp, theta_d)

  # Limit to [-pi/2,pi/2]

  if(np.abs(theta_d) >= np.pi/2. -0.2):
    return 0.,0.,0.
  # theta_d = np.min([np.max([-np.pi/2., theta_d]), np.pi/2.])
  theta_d = np.min([np.pi, theta_d])
  # print(theta_d)
  
  scale = 1.
  theta = copy.deepcopy(theta_d)
  if (theta_d > .00000001):
    for j in range(10):
      # print(theta)
      theta2 = theta*theta
      theta4 = theta2*theta2
      theta6 = theta2*theta4
      theta8 = theta4*theta4

      k0_theta2 = dist[0]*theta2
      k1_theta4 = dist[1]*theta4
      k2_theta6 = dist[2]*theta6
      k3_theta8 = dist[3]*theta8

      theta_fix = ((theta* (1. + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d)/
                          (1. + 3.*k0_theta2 + 5.*k1_theta4 + 7.*k2_theta6 + 9.*k3_theta8))

      theta = theta - theta_fix
      if (np.abs(theta_fix) < .00000001):
        break

      # Old version
      # theta = theta_d / (1.+k0_theta2+k1_theta4+k2_theta6+k3_theta8)

    scale = np.tan(theta) / theta_d

  if np.abs(scale) > 10000:
    print("x: {} \t, y: {} \t, Scale: {}".format(x, y, scale))

  if np.abs(scale) > 500:
    return 0.,0.,0.

  u = xp * scale
  v = yp * scale

  # if u < 0. or u > 960.:
  #   return 0.,0.,0.

  # if v < 0. or v > 960.:
  #   return 0.,0.,0.
  return u,v, scale


def rectifyPoint(x, y, mtx):
  x = x*mtx[0,0] + mtx[0,2]
  y = y*mtx[1,1] + mtx[1,2]
  return x, y


def printExamples(mtx, dist, half_w):
  # xt,yt, rt = undistortPoints(-half_w, 0., dist, mtx, half_w)
  # print("[{}, {}],\t [{}, {}],\t r: {}".format(-half_w, 0., xt, yt, rt))
  # xt,yt, rt = undistortPoints(half_w, 0., dist, mtx, half_w)
  # print("[{}, {}],\t [{}, {}],\t r: {}".format(half_w, 0., xt, yt, rt))
  # xt,yt, rt = undistortPoints(0., -half_w, dist, mtx, half_w)
  # print("[{}, {}],\t [{}, {}],\t r: {}".format(0., -half_w, xt, yt, rt))
  # xt,yt, rt = undistortPoints(0., half_w, dist, mtx, half_w)
  # print("[{}, {}],\t [{}, {}],\t r: {}".format(0., half_w, xt, yt, rt))
  # xt,yt, rt = undistortPoints(-half_w, -half_w, dist, mtx, half_w)
  # print("[{}, {}],\t [{}, {}],\t r: {}".format(-half_w, -half_w, xt, yt, rt))
  # xt,yt, rt = undistortPoints(-10., -10., dist, mtx, half_w)
  # print("[{}, {}],\t [{}, {}],\t r: {}".format(-10., -10., xt, yt, rt))

  xt,yt, rt = undistortPoints(10., 10., dist, mtx, half_w)
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
  xt,yt, rt = distortPoints(10., 10., dist, mtx, half_w)
  print("Ours:\t [{}, {}],\t\t [{:.2f}, {:.2f}],\t r: {:.2f}".format(10., 10., xt, yt, rt))

  print("Theirs:\t [{}, {}],\t [{:.2f}, {:.2f}]".format(point[1,0,0],point[1,0,1], unpoint[1,0,0], unpoint[1,0,1]))
  xt,yt, rt = distortPoints(100., 100., dist, mtx, half_w)
  print("Ours:\t [{}, {}],\t [{:.2f}, {:.2f}],\t r: {:.2f}".format(100., 100., xt, yt, rt))

  print("Theirs:\t [{}, {}],\t [{:.2f}, {:.2f}]".format(point[2,0,0],point[2,0,1], unpoint[2,0,0], unpoint[2,0,1]))
  xt,yt, rt = distortPoints(-200., -200., dist, mtx, half_w)
  print("Ours:\t [{}, {}],\t [{:.2f}, {:.2f}],\t r: {:.2f}".format(-200., -200., xt, yt, rt))  
  print(type(xt), type(yt)) 


def writeToFile(variable, name, options):
  with open(options.folder+"/"+name+".txt", "w") as f:
    np.savetxt(f, variable, fmt='%.2f')


if __name__ == "__main__":
  main() 