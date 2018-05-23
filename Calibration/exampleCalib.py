import sys, time
import numpy as np
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import glob
from optparse import OptionParser
import flattenImages



def main():
  start = time.time()

  parser = OptionParser()  
  parser.add_option("-f", "--folder", dest="folder",
                  help="Choose I/O Folder", metavar="FILE")
  parser.add_option("-w", "--write",
                  action="store_true", dest="writeFile", default=False,
                  help="writes camera parameters to file")
  parser.add_option("-v", "--video",
                  action="store_true", dest="videoInput", default=False,
                  help="choose in input file is a video")
  parser.add_option("-i", "--images",
                  action="store_true", dest="saveImages", default=False,
                  help="select to save images to folder")
  parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")
  parser.add_option("-c", "--crop",
                  action="store_true", dest="halfImg", default=False,
                  help="Choose to use only single fisheye")
  parser.add_option("-s", "--savevid",
                  action="store_true", dest="saveVid", default=False,
                  help="save output as a new video")
  parser.add_option("-e", "--example",
                  action="store_true", dest="showExample", default=False,
                  help="show example image")
  parser.add_option("-b", "--calibrate",
                  action="store_true", dest="calib", default=False,
                  help="perform image calibration")
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
  parser.add_option("--verify",
                  action="store_true", dest="verify", default=False,
                  help="Use user input to verify checkerboard")
  parser.add_option("--fisheye",
                  action="store_true", dest="fisheye", default=False,
                  help="Use fisheye calibration method")
  parser.add_option("--flatten",
                  action="store_true", dest="flatten", default=False,
                  help="Choose whether or not to flatten the images first")  
  parser.add_option("--phone",
                  action="store_true", dest="phone", default=False,
                  help="Decite whether using phone or not")  
  (options, args) = parser.parse_args()


  # checkerboard Dimensions
  cbDims = (8,6)
  # squareSize = 0.022

  # cbDims = (6,7)
  squareSize = 0.12
    
  # termination criteria
  # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  if options.fisheye:
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
  else:
    calibration_flags = cv2.CALIB_FIX_ASPECT_RATIO

  # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
  # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW+cv2.CALIB_USE_INTRINSIC_GUESS

  # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  # objp = np.zeros((cbDims[0]*cbDims[1],3), np.float32)
  # objp[:,:2] = np.mgrid[0:cbDims[1],0:cbDims[0]].T.reshape(-1,2)

  objp = np.zeros((1, cbDims[0]*cbDims[1], 3), np.float32)
  objp[0,:,:2] = np.mgrid[0:cbDims[0], 0:cbDims[1]].T.reshape(-1, 2)

  if(options.videoInput):
    # objpoints, imgpoints, size = calibVideo(folder, cbDims, verbose, start, 
    #                           criteria, objp, halfImg, saveVid, saveImages, calib, saveTime)
    objpoints, imgpoints, size, checkFound = calibVideo(cbDims, start, criteria, objp, options)
  else:
    objpoints, imgpoints, size = calibImages(folder, cbDims, verbose, start, criteria, objp)

  if options.calib:
    if len(objpoints) == 0:
      print("Did not find any corners")
      return
    print("{} Images Found".format(np.sum(checkFound)))

    print "\nCalibrating...",        
    if(options.fisheye):
      N_OK = len(objpoints)
      K = np.eye(3)
      D = np.zeros((4, 1))
      rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
      tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

      ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, size,K,D,rvecs, tvecs, \
                calibration_flags, (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    else:
      ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size,None,None)
    print "Done!\n"

    print("Camera Matrix:")
    print(mtx)
    print("\nDistortion Parameters:")
    print(dist)

    total_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    mean_error = total_error/len(objpoints)

    print("\nRMS Error:\t" + str(round(ret,4)))
    print("Mean Error:\t"+ str(round(mean_error,4)))
    print("Total Error:\t"+ str(round(total_error,4)))

    if(options.writeFile):
      writeToFile(options, ret, mtx, dist, total_error, mean_error)
    if(options.showExample):
      if(options.videoInput):
        k = 0
        for i, j in enumerate(checkFound):
          if j:
            print "Undistorting image: " + str(i)
            dispExample(options, mtx, dist, "image", i, imgpoints[k], cbDims)
            k += 1
      else:
        dispExample(options, mtx, dist, "left")

  cv2.destroyAllWindows()
  print("\nTotal Time:\t" + str(round(time.time()-start,3)))



# def calibVideo(folder, cbDims, verbose, start, criteria, objp, halfImg, saveVid, saveImages, calib, saveTime):
def calibVideo(cbDims, start, criteria, objp, opt):
  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d point in real world space
  imgpoints = [] # 2d points in image plane.

  # vidcap = cv2.VideoCapture(folder+'/*.mp4')
  print "Opening Video...",
  vidcap = cv2.VideoCapture(opt.folder+'/calib.mp4')

  if vidcap.isOpened(): 
    print "Done!\n"
    width   = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height  = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    length  = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = vidcap.get(cv2.CAP_PROP_FPS)
  else:
    print(opt.folder+'/calib.mp4')
    print "ERROR - video not opened"
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
    elif(opt.halfImg):
      M = cv2.getRotationMatrix2D((twidth/2,theight/2),-90,1)
    else:
      print((twidth,theight))
      M = cv2.getRotationMatrix2D((960/2,960/2),-90,1)

  # print twidth, width

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
  if opt.phone:
    ratio = float(twidth)/float(theight)
    scaled_width = int(ratio*960)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 960, 960)

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

    if opt.phone:
      img = cv2.resize(img, (scaled_width, 960))

      width_diff = (scaled_width - 960)/2
      height_diff = (twidth - 960)/2
      img = img[0:960, width_diff:scaled_width-width_diff]

    if opt.rotateImage:
      if opt.phone:
        img = cv2.warpAffine(img,M,(960,960))
      else:
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

      # print(xx.shape)
      # print(grey.shape)
      # print(np.max(xx),np.min(xx))
      for ii, x in enumerate(xx):
        for j, y in enumerate(yy):
          if x**2+y**2 > 1:
            # print(ii, x)
            grey[ii,j] = 0

    if opt.maskImage:
      grey= cv2.bitwise_and(grey,grey,mask = mask)

    if opt.flatten:
      mtx, dist = flattenImages.getCameraMatrix()
      grey = flattenImages.flattenImage(grey, mtx, dist, 1,0)

    if opt.saveVid:
      out.write(grey)

    if opt.saveImages:
      cv2.imwrite("{}/images/image{:d}.jpg".format(opt.folder,numFrames), grey);

    if opt.calib:
      objpoints, imgpoints, checkFound = calibrate(img, grey, objpoints, 
                                          objp, imgpoints, criteria, cbDims, opt.verbose, opt.verify, checkFound)
      print("Time per frame: {:.3f}\t | checkerBoardFound: {""}\t | Total Time: {:.4f}\t | Frame: {:d}\t | Image: {:d}"
                                .format(time.time()-iterTime,checkFound[-1],time.time()-start,i,numFrames))
    else:
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

  print("shape:", grey.shape[::-1])

  return objpoints, imgpoints, grey.shape[::-1], checkFound



def calibImages(folder, cbDims, verbose, start, criteria, objp):
  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d point in real world space
  imgpoints = [] # 2d points in image plane.

  images = glob.glob(folder+'/*.jpg')
  for i, fname in enumerate(images):
    iterTime = time.time()
    img = cv2.imread(fname)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    objpoints, imgpoints, ret = calibrate(img, grey, objpoints, 
                                        objp, imgpoints, criteria, cbDims, verbose)
    if (i%10 == 0 ):
      print("Time per frame: {:.3f}\t | checkerBoardFound: {""}\t | Total Time: {:.4f}\t | Frame: {:d}"
                                .format(time.time()-iterTime,ret,time.time()-start,i))
    else:
      print("Time per frame: {:.3f}\t | checkerBoardFound: {""}\t".format(time.time()-iterTime,ret))
  
  return objpoints, imgpoints, grey.shape[::-1]


def calibrate(img, grey, objpoints, objp, imgpoints, criteria, cbDims, verbose, verify, checkFound):
  
  # Find the chess board corners
  ret, corners = cv2.findChessboardCorners(grey, cbDims,None)

  # If found, add object points, image points (after refining them)
  if ret == True:
      cv2.cornerSubPix(grey,corners,(11,11),(-1,-1),criteria)
            # Draw and display the corners
      if(verify):
        cv2.drawChessboardCorners(img, (cbDims[0],cbDims[1]), corners,ret)
        print("Please press \'y\' to verify corners")
        cv2.imshow('img',img)
        # while True:
        k = cv2.waitKey(0)
          # if(k !=-1):
          #   break
        if k == ord('y'):
          objpoints.append(objp)
          imgpoints.append(corners)
        elif k == ord('q'):
          sys.exit(0)
        else:
          ret = False
          print("Rejected Image!")
        checkFound.append(ret)
        return objpoints, imgpoints, checkFound
      elif(verbose):
        cv2.drawChessboardCorners(img, (cbDims[0],cbDims[1]), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

      objpoints.append(objp)
      imgpoints.append(corners)

  checkFound.append(ret)
  return objpoints, imgpoints, checkFound


def writeToFile(options, ret, mtx, dist, total_error, mean_error):
  with open(options.folder+"/calib_results.txt", "w") as f: 
    f.write("RMS Error:  \t" + str(round(ret,4)) + "\n")
    f.write("Mean Error: \t" + str(round(mean_error,4)) + "\n")
    f.write("Total Error:\t" + str(round(total_error,4)) + "\n\n")
    f.write("Camera Matrix:\n")
    np.savetxt(f, mtx, fmt='%.4f')
    f.write("\nDistortion Parameters:\n")
    np.savetxt(f, dist, fmt='%.6f')

  with open(options.folder+"/calib_yaml.txt", "w") as f:
    f.write("Camera.fx: {:.6f}\n".format(mtx[0,0]))
    f.write("Camera.fy: {:.6f}\n".format(mtx[1,1]))
    f.write("Camera.cx: {:.6f}\n".format(mtx[0,2]))
    f.write("Camera.cy: {:.6f}\n\n".format(mtx[1,2]))

    if(options.fisheye):
      f.write("Camera.k1: {:.6f}\n".format(dist[0,0]))
      f.write("Camera.k2: {:.6f}\n".format(dist[1,0]))
      f.write("Camera.p1: {:.6f}\n".format(dist[2,0]))
      f.write("Camera.p2: {:.6f}\n".format(dist[3,0]))
      f.write("Camera.k3: {:.6f}".format(0.))
    else:
      f.write("Camera.k1: {:.6f}\n".format(dist[0,0]))
      f.write("Camera.k2: {:.6f}\n".format(dist[0,1]))
      f.write("Camera.p1: {:.6f}\n".format(dist[0,2]))
      f.write("Camera.p2: {:.6f}\n".format(dist[0,3]))
      f.write("Camera.k3: {:.6f}".format(dist[0,4]))


def dispExample(opt, mtx, dist, name,i, corners, cbDims):
  img = cv2.imread(opt.folder+'/images/'+name+str(i)+'.jpg', cv2.IMREAD_COLOR)
  h,  w = img.shape[:2]
  
  cv2.drawChessboardCorners(img, (cbDims[0],cbDims[1]), corners,True)

  if(opt.fisheye):
    nmtx = mtx.copy()

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (w,h), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imwrite(opt.folder+'/undistorted/_calibresult'+str(i)+'.jpg',undistorted_img)
    cv2.waitKey(50)
    return  

  newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

  # undistort
  dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
  # crop the image
  # x,y,w,h = roi
  # dst = dst[y:y+h, x:x+w]

  cv2.imwrite(opt.folder+'/undistorted/_calibresult'+str(i)+'.jpg',dst)
  cv2.waitKey(500)
  # print dst.shape

  # # undistort
  # mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
  mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),cv2.CV_32F)
  if(mapx is None or mapy is None):
    print("Error - mapping is None")
    return
  dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

  # # crop the image
  # x,y,w,h = roi
  # dst = dst[y:y+h, x:x+w]
  cv2.imwrite(opt.folder+'/undistorted/calibresult'+str(i)+'.jpg',dst)
  cv2.waitKey(50)


if __name__ == "__main__":
  main() 