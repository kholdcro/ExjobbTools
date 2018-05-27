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
  parser.add_option("-i", "--images",
                  action="store_true", dest="saveImages", default=False,
                  help="select to save images to folder")
  parser.add_option("-c", "--crop",
                  action="store_true", dest="halfImg", default=False,
                  help="Choose to use only single fisheye")
  parser.add_option("-s", "--savevid",
                  action="store_true", dest="saveVid", default=False,
                  help="save output as a new video")
  parser.add_option("-t", "--saveTime",
                  action="store_true", dest="saveTime", default=False,
                  help="saves timestamps to txt file")
  parser.add_option("-r", "--resize",
                  action="store_true", dest="resize", default=False,
                  help="halfs image width and height")
  parser.add_option("-m", "--mask",
                  action="store_true", dest="maskImage", default=False,
                  help="masks the images of the drone")
  parser.add_option("-a", "--rotate",
                  action="store_true", dest="rotateImage", default=False,
                  help="rotates the images of the drone")
  parser.add_option("--startFrame", dest="startFrame",
                  help="choose frame to start on", type="int", default=0)
  parser.add_option("--subFrames", dest="subsampleFrames",
                  help="choose every x frames to evaluate", type="int", default=1)
  parser.add_option("--right",
                  action="store_true", dest="rightImage", default=False,
                  help="Uses right fisheye")
  parser.add_option("-p", "--pseudoSmall",
                  action="store_true", dest="pseudoSmall", default=False,
                  help="Remaps fisheye to smaller FOV")
  (options, args) = parser.parse_args()

  mtx, dist = getCameraMatrix()
  exctractVideo(options, mtx, dist, start)

  cv2.destroyAllWindows()
  print("\nTotal Time:\t" + str(round(time.time()-start,3)))


def getCameraMatrix():
  mtx = np.zeros((3,3))
  mtx[0,0] = 301.887922
  mtx[1,1] = 301.720198
  mtx[0,2] = 481.204306
  mtx[1,2] = 478.820797
  mtx[2,2] = 1.

  dist = np.array([-0.017766, 0.000774, 0.007102, -0.003659], dtype=np.float32)
  return mtx, dist


# def calibVideo(folder, cbDims, verbose, start, criteria, objp, halfImg, saveVid, saveImages, calib, saveTime):
def exctractVideo(opt, mtx, dist, start):
  # vidcap = cv2.VideoCapture(folder+'/*.mp4')
  print("Opening Video...",)
  vidcap = cv2.VideoCapture(opt.folder+'/calib.mp4')

  if vidcap.isOpened(): 
    print("Done!\n")
    width   = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height  = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    length  = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = vidcap.get(cv2.CAP_PROP_FPS)
  else:
    print("ERROR - video not opened")
    sys.exit()

  whalf = width
  if opt.halfImg or opt.rightImage:
    whalf = int(round(width/2))

  twidth = whalf
  theight = height

  if opt.resize:
    twidth /= 2
    theight /= 2

  if opt.rotateImage:
    if(opt.rightImage):
      M = cv2.getRotationMatrix2D((twidth/2,theight/2),90,1)
    else:
      M = cv2.getRotationMatrix2D((twidth/2,theight/2),-90,1)

  if opt.saveVid:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'x264' doesn't work
    out = cv2.VideoWriter('{}/saved.avi'.format(opt.folder), fourcc, fps, (theight,twidth), False)

  print("Video length: {:d} frames\n".format(length))

  if opt.maskImage:
    mask = cv2.imread(opt.folder+'/mask.jpg')
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    if opt.resize:
      mask = cv2.resize(mask, (theight,twidth))
      mask_inv = cv2.resize(mask_inv, (theight,twidth))

  i = 0
  numFrames = 0
  success = True
  checkFound = []
  # while(vidcap.isOpened()) and i < 5:
  while(vidcap.isOpened()):
    iterTime = time.time()
    success,img = vidcap.read()

    if i < opt.startFrame or i%opt.subsampleFrames != 0:
      i += 1
      continue
    
    if not success:
      print("\nImage read was not success\n")
      break

    if opt.halfImg:
      img = img[0:height, 0:whalf]
      # img = img[0:height, whalf:2*whalf]
    
    if opt.rightImage:
      img = img[0:height, whalf:2*whalf]

    if opt.resize:
      img = cv2.resize(img, (twidth, theight))

    if opt.rotateImage:
      img = cv2.warpAffine(img,M,(theight,twidth))

    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if opt.pseudoSmall:
      center = grey.shape[0]/2
      quarter = center/2
      grey = grey[quarter:center+quarter, quarter:center+quarter]
      img = img[quarter:center+quarter, quarter:center+quarter]
      xx = np.linspace(grey.shape[0]/2,-grey.shape[0]/2, num=grey.shape[0])
      xx = 2*xx/grey.shape[0]
      yy = np.linspace(grey.shape[1]/2,-grey.shape[1]/2, num=grey.shape[1])
      yy = 2*yy/grey.shape[1]

      for ii, x in enumerate(xx):
        for j, y in enumerate(yy):
          if x**2+y**2 > 1:
            grey[ii,j] = 0

    if opt.maskImage:
      grey= cv2.bitwise_and(grey,grey,mask = mask)

    grey = flattenImage(grey, mtx, dist, 2,0)
    grey = cv2.resize(grey, (twidth, theight))

    if opt.saveVid:
      out.write(grey)

    if opt.saveImages:
      cv2.imwrite("{}/images/image{:d}.jpg".format(opt.folder,numFrames), grey);

    print("Time per frame: {:.3f}\t | Total Time: {:.4f}\t | Frame: {:d}"
                                  .format(time.time()-iterTime,time.time()-start,i))
    imgSize = grey.shape[::-1]
    i += 1
    numFrames += 1

  vidcap.release()
  if opt.saveVid:
    out.release()
  
  if opt.saveTime:
    timestep = 1/fps
    with open(opt.folder+"/times.txt", "w") as f:
      for i in range(numFrames):
        f.write("{:e}\n".format(timestep*i))

  return


def flattenImage(img, mtx, dist, size, balance=1.):
  h,  w = img.shape[:2]
  half_w = int(w/2)

  dim1 = img.shape[:2]  #dim1 is the dimension of input image to un-distort

  dim2 = tuple(np.multiply(size,dim1))
  dim3 = tuple(np.multiply(size,dim1))

  m_K = mtx.copy()
  # new_K = m_K*size
  # new_K[2][2] = 1.0
  new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(m_K, dist, dim2, np.eye(3), balance=balance)
  map1, map2 = cv2.fisheye.initUndistortRectifyMap(m_K, dist, np.eye(3), new_K, dim3, cv2.CV_16SC2)
  undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


  return undistorted_img


if __name__ == "__main__":
  main() 