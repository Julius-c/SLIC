# Simple Linear Iterative Clustering
import numpy as np
import sys
import cv2
from math import *

class SLIC:
	def __init__(self, image, supervoxels, nc):
		rgb2lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		self.rgb2lab = np.array([rgb2lab, rgb2lab, rgb2lab, rgb2lab, rgb2lab])
		self.image = np.array([image, image, image, image, image])
		self.depth = self.rgb2lab.shape[0]
		self.height = self.rgb2lab.shape[1]
		self.width = self.rgb2lab.shape[2]
		self.voxels = self.depth * self.height * self.width
		self.supervoxels = supervoxels
		self.step = int(pow(self.voxels / supervoxels, 1. / 3))
		self.nc = nc
		self.ns = self.step

		self.MAX = 1e9
		self.iterations = 10

		self.labels = -1 * np.ones(self.image.shape[:3])
		self.distances = self.MAX * np.ones(self.image.shape[:3])

	def findLowestGrad(self, center):
		lowGrad = self.MAX
		localMin = center
		for z in xrange(center[0] - 1, center[0] + 2):
			for y in xrange(center[1] - 1, center[1] + 2):
				for x in xrange(center[2] - 1, center[2] + 2):
					l1 = self.rgb2lab[z, y, x][0]
					l2 = self.rgb2lab[z, y + 1, x][0]
					l3 = self.rgb2lab[z, y, x + 1][0]
					curGrad = (l1 - l2 if l1 > l2 else l2 - l1) + (l1 - l3 if l1 > l3 else l3 - l1)
					if (curGrad < lowGrad):
						lowGrad = curGrad
						localMin = [z, y, x]
		return localMin

	def computeDist(self, index, point):
		color = self.rgb2lab[point[0], point[1], point[2]]
		dc = sqrt(pow(self.centers[index][0] - color[0], 2)
					+ pow(self.centers[index][1] - color[1], 2)
					+ pow(self.centers[index][2] - color[2], 2))
		ds = sqrt(pow(self.centers[index][3] - point[0], 2)
					+ pow(self.centers[index][4] - point[1], 2)
					+ pow(self.centers[index][5] - point[2], 2))
		return sqrt(pow(dc / self.nc, 2) + pow(ds / self.ns, 2))

	def initialize(self):
		centers = []
		nzs = self.depth / self.step
		if nzs <= 0:
			nzs = 1
		zs = self.depth / nzs
		if zs >= self.depth:
			zs = 0
		zoff = self.step / 2 if self.step < self.depth else self.depth / 2
		for zi in xrange(nzs):
			z = zoff + zi * zs
			if z >= self.depth:
				z = self.depth - zoff
			for y in xrange(self.step / 2, self.height - self.step / 2, self.step):
				for x in xrange(self.step / 2, self.width - self.step / 2, self.step):
					oc = [z, y, x]
					nc = self.findLowestGrad(oc)
					ncolor = self.rgb2lab[nc[0], nc[1], nc[2]]
					centers.append([ncolor[0], ncolor[1], ncolor[2], nc[0], nc[1], nc[2]])
		self.centers = np.array(centers)

	def generateSuperVoxels(self):
		for i in xrange(self.iterations):
			self.distances = self.MAX * np.ones(self.image.shape[:3])
			for j in xrange(len(self.centers)):
				for z in xrange(self.centers[j][3] - self.step, self.centers[j][3] + self.step):
					for y in xrange(self.centers[j][4] - self.step, self.centers[j][4] + self.step):
						for x in xrange(self.centers[j][5] - self.step, self.centers[j][5] + self.step):
							if (z >= 0 and z < self.depth \
								and y >= 0 and y < self.height \
								and x >= 0 and x < self.width):
								point = [z, y, x]
								dist = self.computeDist(j, point)
								if (dist < self.distances[z, y, x]):
									self.distances[z, y, x] = dist
									self.labels[z, y, x] = j
			# update center
			for j in xrange(len(self.centers)):
				jcluster = self.labels == j
				self.centers[j][0:3] = np.sum(self.rgb2lab[jcluster], axis=0)
				self.centers[j][3:] = [0, 0, 0]
			for z in xrange(self.depth):
				for y in xrange(self.height):
					for x in xrange(self.width):
						label = int(self.labels[z, y, x])
						self.centers[label][3:] += [z, y, x]
			for j in xrange(len(self.centers)):
				jcluster = self.labels == j
				self.centers[j] /= np.sum(jcluster)

	def enforceLabelConnectivity(self):
		label = 0
		adjlabel = 0
		new_labels = -1 * np.ones(self.image.shape[:3])
		ideal = self.voxels / len(self.centers)
		# left up right down back front
		dx4 = [-1, 0, 1, 0, 0, 0]
		dy4 = [0, -1, 0, 1, 0, 0]
		dz4 = [0, 0, 0, 0, 1, -1]
		for k in xrange(self.depth):
			for j in xrange(self.height):
				for i in xrange(self.width):
					if new_labels[k, j, i] == -1:
						members = [(k, j, i)]
						for dx, dy, dz in zip(dx4, dy4, dz4):
							x = i + dx
							y = j + dy
							z = k + dz
							if (x >= 0 and x < self.width and \
								y >= 0 and y < self.height and \
								z >= 0 and z < self.depth and \
								new_labels[z, y, x] >= 0):
								adjlabel = new_labels[z, y, x]
						count = 1
						c = 0
						while c < count:
							for dx, dy, dz in zip(dx4, dy4, dz4):
								x = members[c][2] + dx
								y = members[c][1] + dy
								z = members[c][0] + dz 
								if (x >= 0 and x < self.width and \
									y >= 0 and y < self.height and \
									z >= 0 and z < self.depth):
									if (new_labels[z, y, x] == -1 and self.labels[z, y, x] == self.labels[k, j, i]):
										members.append((z, y, x))
										new_labels[z, y, x] = label
										count += 1
							c += 1
						if (count <= ideal >> 2):
							for c in xrange(count):
								new_labels[members[c]] = adjlabel
							label -= 1
						label += 1
		self.labels = new_labels

	def displayContour(self, color):
		# 3 * 3 cube
		dz26 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1]
		dx26 = [-1, -1, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1, 0, 0]
		dy26 = [0, -1, -1, -1, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0, 0]

		isTaken = np.zeros(self.image.shape[:3], np.bool)
		contours = []

		for k in xrange(self.depth):
		    for j in xrange(self.height):
					for i in xrange(self.width):
						count = 0
						for dx, dy, dz in zip(dx26, dy26, dz26):
							x = i + dx
							y = j + dy
							z = k + dz
							if x >= 0 and x < self.width \
								and y >= 0 and y < self.height \
								and z >= 0 and z < self.depth:
								if isTaken[z, y, x] == False and self.labels[k, j, i] != self.labels[z, y, x]:
									count += 1

						if count >= 8:
							isTaken[k, j, i] = True
							contours.append([k, j, i])

		for i in xrange(len(contours)):
		    self.image[contours[i][0], contours[i][1], contours[i][2]] = color

	def run(self):
		self.initialize()
		self.generateSuperVoxels()
		self.enforceLabelConnectivity()
		self.displayContour((0, 0, 255))
		for i in xrange(self.depth):
			s = "slic" + str(i) + ".jpg"
			cv2.imwrite(s, self.image[i])


image = cv2.imread(sys.argv[1])
supervoxels = int(sys.argv[2])
nc = int(sys.argv[3])
slic = SLIC(image, supervoxels, nc)
slic.run()
