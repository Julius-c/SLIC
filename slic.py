# Simple Linear Iterative Clustering
import numpy as np
import sys
import cv2
from math import *

class SLIC:
	def __init__(self, image, superpixels, nc):
		self.image = image
		self.height = image.shape[0]
		self.width = image.shape[1]
		self.pixels = self.height * self.width
		self.superpixels = superpixels
		self.step = int(sqrt(self.pixels / superpixels))
		self.nc = nc
		self.ns = self.step
		self.rgb2lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

		self.MAX = 1e9
		self.iterations = 10

		self.labels = -1 * np.ones(self.image.shape[:2])
		self.distances = self.MAX * np.ones(self.image.shape[:2])

	def findLowestGrad(self, center):
		lowGrad = self.MAX
		localMin = center
		for i in xrange(center[0] - 1, center[0] + 2):
			for j in xrange(center[1] - 1, center[1] + 2):
				l1 = self.rgb2lab[j, i][0]
				l2 = self.rgb2lab[j, i + 1][0]
				l3 = self.rgb2lab[j + 1, i][0]
				curGrad = (l1 - l2 if l1 > l2 else l2 - l1) + (l1 - l3 if l1 > l3 else l3 - l1)
				if (curGrad < lowGrad):
					lowGrad = curGrad
					localMin = [i, j]
		return localMin

	def computeDist(self, index, point):
		color = self.rgb2lab[point[1], point[0]]
		dc = sqrt(pow(self.centers[index][0] - color[0], 2)
					+ pow(self.centers[index][1] - color[1], 2)
					+ pow(self.centers[index][2] - color[2], 2))
		ds = sqrt(pow(self.centers[index][3] - point[0], 2)
					+ pow(self.centers[index][4] - point[1], 2))
		return sqrt(pow(dc / self.nc, 2) + pow(ds / self.ns, 2))

	def initialize(self):
		centers = []
		for i in xrange(self.step, self.width - self.step / 2, self.step):
			for j in xrange(self.step, self.height - self.step / 2, self.step):
				oc = [i, j]
				nc = self.findLowestGrad(oc)
				ncolor = self.rgb2lab[nc[1], nc[0]]
				centers.append([ncolor[0], ncolor[1], ncolor[2], nc[0], nc[1]])
		self.centers = np.array(centers)

	def generateSuperPixels(self):
		coordinate = np.mgrid[0:self.height, 0:self.width].swapaxes(0,2).swapaxes(0,1)
		for i in xrange(self.iterations):
			self.distances = self.MAX * np.ones(self.image.shape[:2])
			for j in xrange(len(self.centers)):
				for m in xrange(self.centers[j][3] - self.step, self.centers[j][3] + self.step):
					for n in xrange(self.centers[j][4] - self.step, self.centers[j][4] + self.step):
						if (m >= 0 and m < self.width and n >= 0 and n < self.height):
							point = [m, n]
							dist = self.computeDist(j, point)
							if (dist < self.distances[n, m]):
								self.distances[n, m] = dist
								self.labels[n, m] = j
			# update center
			for j in xrange(len(self.centers)):
				jcluster = self.labels == j
				self.centers[j][0:3] = np.sum(self.rgb2lab[jcluster], axis=0)
				sumy, sumx = np.sum(coordinate[jcluster], axis=0)
				self.centers[j][3:] = sumx, sumy
				self.centers[j] /= np.sum(jcluster)

	def enforceLabelConnectivity(self):
		label = 0
		adjlabel = 0
		new_labels = -1 * np.ones(self.image.shape[:2])
		ideal = self.width * self.height / len(self.centers)
		# left up right down
		dx4 = [-1, 0, 1, 0]
		dy4 = [0, -1, 0, 1]
		for j in xrange(self.height):
			for i in xrange(self.width):
				if new_labels[j, i] == -1:
					members = [(j, i)]
					for dx, dy in zip(dx4, dy4):
						x = i + dx
						y = j + dy
						if (x >= 0 and x < self.width and \
							y >= 0 and y < self.height and \
							new_labels[y, x] >= 0):
							adjlabel = new_labels[y, x]
					count = 1
					c = 0
					while c < count:
						for dx, dy in zip(dx4, dy4):
							x = members[c][1] + dx
							y = members[c][0] + dy
							if (x >= 0 and x < self.width and y >= 0 and y < self.height):
								if (new_labels[y, x] == -1 and self.labels[y, x] == self.labels[j, i]):
									members.append((y, x))
									new_labels[y, x] = label
									count += 1
						c += 1
					if (count <= ideal >> 2):
						for c in xrange(count):
							new_labels[members[c]] = adjlabel
						label -= 1
					label += 1
		self.labels = new_labels

	def displayContour(self, color):
		# 3 * 3 square
		dx8 = [-1, -1, 0, 1, 1, 1, 0, -1]
		dy8 = [0, -1, -1, -1, 0, 1, 1, 1]

		isTaken = np.zeros(self.image.shape[:2], np.bool)
		contours = []

		for i in xrange(self.width):
		    for j in xrange(self.height):
		        count = 0
		        for dx, dy in zip(dx8, dy8):
		            x = i + dx
		            y = j + dy
		            if x >= 0 and x < self.width and y >= 0 and y < self.height:
		                if isTaken[y, x] == False and self.labels[j, i] != self.labels[y, x]:
		                    count += 1

		        if count >= 2:
		            isTaken[j, i] = True
		            contours.append([j, i])

		for i in xrange(len(contours)):
		    self.image[contours[i][0], contours[i][1]] = color

	def run(self):
		self.initialize()
		self.generateSuperPixels()
		self.enforceLabelConnectivity()
		self.displayContour((0, 0, 255))
		cv2.imwrite("slic.jpg", self.image)


image = cv2.imread(sys.argv[1])
superpixels = int(sys.argv[2])
nc = int(sys.argv[3])
slic = SLIC(image, superpixels, nc)
slic.run()
