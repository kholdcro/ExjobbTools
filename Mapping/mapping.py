import sys, time
import numpy as np
import cv2
import glob
from optparse import OptionParser
from pyntcloud import PyntCloud
import pandas as pd
from IPython.display import IFrame

def main():
	start = time.time()

	parser = OptionParser()  
	parser.add_option("-i", "--image", dest="image",
	              help="Choose I/O Folder", metavar="FILE")

	(options, args) = parser.parse_args()

	img = cv2.imread(options.image)

	if img is None:
		print("Image Not Found at: {}".format(options.image))
		return

	hgt = img.shape[0]
	wth = img.shape[1]
	center = int(wth/2)

	circlePx = int(906/2)
	lengthPixels = np.linspace(circlePx-1, 0, num=circlePx, dtype=int)

	circleSqr = circlePx**2
	
	borders = np.zeros((center,center), dtype = np.uint8)
	
	ii = 0
	for x in reversed(lengthPixels):
		x2 = x**2
		y = lengthPixels[ii]
		y2 = y**2
		while(x2+y2 > circleSqr):
			ii += 1
			y = lengthPixels[ii]
			y2 = y**2
		borders[x][:y] = 1


	# angVec = lengthPixels*.0022
	# angVec = lengthPixels*.0022
	angVec = lengthPixels*(np.pi/(2*circlePx))

	angles = np.zeros((center, center), dtype = np.float32)

	for x in range(center):
		for y in range(center):
			if borders[x][y] > 0:
				angles[y][x] = angVec[circlePx-1-x]

	borders = np.concatenate((np.flip(borders,axis=0),borders), axis=0)
	borders = np.concatenate((np.flip(borders,axis=1),borders), axis=1)

	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	grey= grey*borders

	# cv2.imshow("borders", borders)
	# cv2.waitKey(25)   
	# cv2.imshow("mask", mask)
	# cv2.waitKey(25)   
	cv2.imshow("grey", grey)
	cv2.waitKey(1000000)    

	angles = np.concatenate((np.flip(angles,axis=0),angles), axis=0)
	angles = np.concatenate((np.flip(-angles,axis=1),angles), axis=1)

	theta = angles
	phi = np.flip(angles.T, axis=0)

	with open("theta.txt", "w") as f: 
	    f.write("Theta  \n")	
	    np.savetxt(f, theta, fmt='%.3f')

	with open("phi.txt", "w") as f: 
		f.write("Phi  \n")	
		np.savetxt(f, phi, fmt='%.3f')

	print(img.shape)
	print(np.max(phi))
	print(np.min(phi))
	print(phi.shape)

	# cv2.imshow("phi", cv2.cvtColor((phi+1)/2, cv2.COLOR_GRAY2BGR))
	# cv2.waitKey(10)
	# cv2.imshow("theta", cv2.cvtColor((theta+1)/2, cv2.COLOR_GRAY2BGR))
	# cv2.waitKey(25)
	# cv2.imwrite("coooooollll.jpg", cv2.cvtColor((theta+1)/2, cv2.COLOR_GRAY2BGR));
	# cv2.waitKey(25)


	# theta_c = np.zeros(borders.shape, dtype = np.float32)
	# for i in range(wth):
	# 	for j in range(wth):
	# 		if borders[i,j] > 0:
	# 			print(theta[i,j], np.cos(theta[i,j]))
	# 			theta_c[i,j] = np.cos(theta[i,j])

	theta_s = np.sin(theta)
	theta_c = np.cos(theta)
	phi_s = np.sin(phi)
	phi_c = np.cos(phi)

	# cv2.imshow("theta_s", cv2.cvtColor((theta_s), cv2.COLOR_GRAY2BGR))
	# cv2.waitKey(10)
	# cv2.imshow("theta_c", cv2.cvtColor((theta_c), cv2.COLOR_GRAY2BGR))
	# cv2.waitKey(25)

	x = theta_s*phi_c
	y = theta_s*phi_s
	z = theta_c

	count = np.count_nonzero(borders)
	points = np.empty((count,3))
	colours = np.empty((count,3))

	k = 0
	for i,t in enumerate(borders!=0):
		h = ([j for j, h in enumerate(t) if h])
		for j in h:
			points[k] = [x[i,j],y[i,j],z[i,j]]
			k += 1

	with open("points.txt", "w") as f: 
		f.write("points  \n")	
		np.savetxt(f, points, fmt='%.3f')

	print(points.shape)

	k = 0
	for i,t in enumerate(borders!=0):
		h = ([j for j, h in enumerate(t) if h])
		for j in h:
			colours[k] = img[i,j]
			k += 1

	# print(colours)
	print("StartingLoop\n")
	# cv2.waitKey(100)

def imgToSphere():
	return

def sphereToPointCloud():
	return


if __name__ == "__main__":
	main() 