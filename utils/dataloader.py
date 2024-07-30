import numpy as np
import os
import torch
import torch.utils.data
import h5py
from pysdf import SDF
import time
import open3d as o3d


class DataLoader(torch.utils.data.Dataset):
	def __init__(
		self,
		test_flag,
		inpt = None,
	):
		start_time = time.time()
		self.inpt = inpt




		self.N = 64



		voxels_low = np.zeros([self.N,self.N,self.N,3])
		for i in range(self.N):
			for j in range(self.N):
				for k in range(self.N):
					voxels_low[i,j,k] = [i,j,k]
		self.voxels_low = voxels_low
		
		

		
		voxels_high = np.zeros([self.N*4,self.N*4,self.N*4,3])	
		for i in range(self.N*4):
			for j in range(self.N*4):
				for k in range(self.N*4):
					voxels_high[i,j,k] = [i,j,k]
		self.voxels_high = voxels_high

		self.mesh = o3d.io.read_triangle_mesh(os.path.join('samples',self.inpt))
		f = SDF(np.asarray(self.mesh.vertices), np.asarray(self.mesh.triangles))


		mesh_vertices = np.asarray(self.mesh.vertices)
		

		max_corner = np.amax(mesh_vertices, axis=0)
		min_corner = np.amin(mesh_vertices, axis=0)
		
		all_points_low = self.voxels_low.reshape(-1,3)
		all_points_low = all_points_low/self.voxels_low.shape[0]
		
		all_points_low[:,0] = (max_corner[0]-min_corner[0])*all_points_low[:,0]+(min_corner[0])
		all_points_low[:,1] = (max_corner[1]-min_corner[1])*all_points_low[:,1]+(min_corner[1])
		all_points_low[:,2] = (max_corner[2]-min_corner[2])*all_points_low[:,2]+(min_corner[2])

		indices_low = f.contains(all_points_low)


		indices_low = -2*indices_low + 1  


		self.indices_low = indices_low
		self.all_points_low = all_points_low


		###############################################


		all_points_high = self.voxels_high.reshape(-1,3)
		all_points_high = all_points_high/self.voxels_high.shape[0]
		
		all_points_high[:,0] = (max_corner[0]-min_corner[0])*all_points_high[:,0]+(min_corner[0])
		all_points_high[:,1] = (max_corner[1]-min_corner[1])*all_points_high[:,1]+(min_corner[1])
		all_points_high[:,2] = (max_corner[2]-min_corner[2])*all_points_high[:,2]+(min_corner[2])








		self.all_points_high = all_points_high

		self.dimension = max_corner - min_corner
		print(time.time()-start_time)
	def __len__(self):
		return 1000
		#return len(self.data_names)

	def __getitem__(self, inpt):	
		
	
		return self.indices_low.astype(np.float32) , self.all_points_low.astype(np.float32),self.all_points_high.astype(np.float32), self.dimension, inpt


