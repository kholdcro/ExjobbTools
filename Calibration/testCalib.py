import sys, time
import numpy as np
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import glob
from optparse import OptionParser



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
  img = cv2.imread(opt.folder+'/test.jpg', cv2.IMREAD_COLOR)
  h,  w = img.shape[:2]

  balance=1.
  # balance=0.0
  dim2=None
  dim3=None

  dim1 = img.shape[:2]  #dim1 is the dimension of input image to un-distort

  dim2 = tuple(np.multiply(1,dim1))
  dim3 = tuple(np.multiply(1,dim1))

  edge = 906
  diff = w-edge

  point = np.zeros((4,1,2), dtype=np.float32)
  point[0] = [w/2, edge]
  point[1] = [w/2, diff]
  point[2] = [diff, h/2]
  point[3] = [edge, h/2]

  unpoint = cv2.fisheye.undistortPoints(point,mtx.copy(),dist, np.eye(3),mtx.copy())

  print("Point, \t Undistorted")
  for i in range(len(point)):
    print("[{}, {}] \t [{}, {}]".format(point[i,0,0],point[i,0,1],unpoint[i,0,0],unpoint[i,0,1]))

  minY = unpoint[1,0,1]
  maxY = unpoint[0,0,1]
  minX = unpoint[2,0,0]
  maxX = unpoint[3,0,0]

  print(minY,maxY,minX,maxX)

  xDim = maxX+np.abs(minX)
  yDim = maxY+np.abs(minY)

  mDim = np.maximum(xDim, yDim)
  # dim2 = (mDim, mDim)
  print(dim2)
  # dim3 = dim2

  m_K = mtx.copy()

  # dist = np.array([0,0,0,0], dtype=np.float32)
  new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(m_K, dist, dim2, np.eye(3), balance=balance)
  # new_K = mtx.copy()
  # new_K[0,2] = xDim/2
  # new_K[1,2] = yDim/2
  map1, map2 = cv2.fisheye.initUndistortRectifyMap(m_K, dist, np.eye(3), new_K, dim3, cv2.CV_16SC2)
  undistorted_img2 = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
  # undistorted_img2 = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

  point = np.zeros((7,1,2), dtype=np.float32)
  point[0] = [w/4, h-h/4]
  point[1] = [w/2, diff]
  point[2] = [diff, h/2]
  point[3] = [edge, h/2]
  point[4] = [w/2, edge]
  point[5] = [w/4, h/4]
  point[6] = [w-w/3, h-h/2.5]

  print("mtx, newK")
  print(mtx)
  print(new_K)
  # dist = np.array([0,0,0,0], dtype=np.float32)
  
  # unpoint = cv2.fisheye.undistortPoints(point,mtx,dist, R=np.eye(3),P=mtx)
  unpoint = cv2.fisheye.undistortPoints(point,m_K,dist, np.eye(3),new_K)

  print("Point, \t Undistorted")
  for i in range(len(point)):
    print("[{}, {}] \t [{}, {}]".format(point[i,0,0],point[i,0,1],unpoint[i,0,0],unpoint[i,0,1]))

  grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  ret, corners = cv2.findChessboardCorners(grey, cbDims,None)
  
  cv2.drawChessboardCorners(img, (cbDims[0],cbDims[1]), corners,True)

  nmtx = mtx.copy()

  map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, R=np.eye(3), P=mtx , size=(w,h), m1type=cv2.CV_16SC2)
  undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

  # undistorted_img2 = cv2.fisheye.undistortImage(img, m_K, dist, None, new_K)

  i = 0
  cv2.circle(img,(int(point[i,0,0]),int(point[i,0,1])), 20, (0,0,255), 4)
  cv2.circle(undistorted_img2,(int(unpoint[i,0,0]),int(unpoint[i,0,1])), 20, (0,0,255), 4)

  i += 1
  cv2.circle(img,(int(point[i,0,0]),int(point[i,0,1])), 20, (0,255,255), 4)
  cv2.circle(undistorted_img2,(int(unpoint[i,0,0]),int(unpoint[i,0,1])), 20, (0,255,255), 4)

  i += 1
  cv2.circle(img,(int(point[i,0,0]),int(point[i,0,1])), 20, (255,0,255), 4)
  cv2.circle(undistorted_img2,(int(unpoint[i,0,0]),int(unpoint[i,0,1])), 20, (255,0,255), 4)

  i += 1
  cv2.circle(img,(int(point[i,0,0]),int(point[i,0,1])), 20, (255,0,0), 4)
  cv2.circle(undistorted_img2,(int(unpoint[i,0,0]),int(unpoint[i,0,1])), 20, (255,0,0), 4)

  i += 1
  cv2.circle(img,(int(point[i,0,0]),int(point[i,0,1])), 20, (255,255,0), 4)
  cv2.circle(undistorted_img2,(int(unpoint[i,0,0]),int(unpoint[i,0,1])), 20, (255,255,0), 4)

  i += 1
  cv2.circle(img,(int(point[i,0,0]),int(point[i,0,1])), 20, (0,255,0), 4)
  cv2.circle(undistorted_img2,(int(unpoint[i,0,0]),int(unpoint[i,0,1])), 20, (0,255,0), 4)

  i += 1
  cv2.circle(img,(int(point[i,0,0]),int(point[i,0,1])), 20, (255,255,255), 4)
  cv2.circle(undistorted_img2,(int(unpoint[i,0,0]),int(unpoint[i,0,1])), 20, (255,255,255), 4)

  cv2.imshow("regular", img)
  cv2.waitKey(50)
  cv2.imshow("undistorted", undistorted_img)
  cv2.waitKey(50)
  cv2.imwrite(opt.folder+'/undistortedtest2.jpg',undistorted_img2)
  cv2.waitKey(50)
  cv2.imwrite(opt.folder+'/dottest.jpg',img)
  cv2.waitKey(50)
  cv2.imwrite(opt.folder+'/undistortedtest.jpg',undistorted_img)
  cv2.waitKey(1)
  cv2.waitKey(50)
  cv2.destroyAllWindows()


  newSize = [100, 100]
  rview = np.zeros(newSize)
  K = mtx
  D = dist

  newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, img.shape[:2], np.eye(3), balance=1)

  map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, img.shape[:2], cv2.CV_16SC2)

  rview = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, rview);

  cv2.imshow("Image View", img);
  cv2.waitKey(1)
  cv2.imshow("output", rview);
  cv2.waitKey(1)
  cv2.imwrite(opt.folder+'/output.jpg',rview)
  cv2.waitKey(1)
  cv2.waitKey(0)

  writeToFile(opt, new_K, dist)


def writeToFile(options, new_k, dist):
  with open(options.folder+"/newK.txt", "w") as f:
    f.write("newK.fx: {:.6f}\n".format(new_k[0,0]))
    f.write("newK.fy: {:.6f}\n".format(new_k[1,1]))
    f.write("newK.cx: {:.6f}\n".format(new_k[0,2]))
    f.write("newK.cy: {:.6f}\n\n".format(new_k[1,2]))

    f.write("Camera.k1: {:.6f}\n".format(dist[0]))
    f.write("Camera.k2: {:.6f}\n".format(dist[1]))
    f.write("Camera.p1: {:.6f}\n".format(dist[2]))
    f.write("Camera.p2: {:.6f}\n".format(dist[3]))
    f.write("Camera.k3: {:.6f}".format(0.))


if __name__ == "__main__":
  main() 