import os
import time
import torch
import mcubes
from utils import utils
import numpy as np
import open3d as o3d
import pymesh

def extract_largest_component(mesh):
    # Separate the mesh into connected components
    components = pymesh.separate_mesh(mesh)

    # Find the largest component based on the number of vertices
    largest_mesh = max(components, key=lambda component: component.num_vertices)

    return largest_mesh

def create_mesh_mc(
	all_points, current, dimension, filename, N=256, max_batch=260**3, threshold=0.5
):
	"""
	Create a mesh using the marching cubes algorithm.

	Args:
		generator: The generator of network.
		shape_3d: 3D shape parameters.
		shape_code: Shape code.
		N: Resolution parameter.
		threshold: Marching cubes threshold value.
	"""
	start = time.time()
	mesh_filename = filename
 
	voxel_origin = [0, 0, 0]
	dimension = dimension.cpu().numpy()[0]
	voxel_size = np.asarray([N+4,N+4,N+4]).astype(int)


	voxels = np.repeat(np.expand_dims(np.zeros(voxel_size),-1),4,axis=-1)
	for i in range(voxel_size[0]):
		for j in range(voxel_size[1]):
			for k in range(voxel_size[2]):
				voxels[...,:3][i,j,k] = [i,j,k]
	
	voxels[...,:3] = voxels[...,:3]*np.expand_dims(np.expand_dims(np.expand_dims(dimension, 0),0),0)/N
	samples = torch.from_numpy(voxels)


	num_samples = np.prod(voxel_size)

	samples.requires_grad = False

	head = 0
	occ = torch.zeros_like(torch.from_numpy(voxels[...,0])).unsqueeze(0)
	print(occ.shape)
	print(current.shape)
	occ[:,2:-2, 2:-2, 2:-2] = current.reshape(1,N,N,N)
	samples = samples.reshape([-1,4])
	occ = occ.reshape([-1,1])
	while head < num_samples:
		sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

	# 	occ,_,_ = generator(sample_subset.unsqueeze(0), shape_3d, shape_code)
		samples[head : min(head + max_batch, num_samples), 3] = (
			occ[:,head : min(head + max_batch, num_samples)].reshape(-1)
			.detach()
			.cpu()
		)
		head += max_batch

	sdf_values = samples[:, 3]
	sdf_values = sdf_values.reshape(N+4, N+4, N+4)
			
	end = time.time()
	print(f"Sampling took: {end - start:.3f} seconds")

	numpy_3d_sdf_tensor = sdf_values.numpy()

	verts, faces = mcubes.marching_cubes(numpy_3d_sdf_tensor, threshold)
	
	mesh_points = verts
	mesh_points = (mesh_points + 0.5) / (N+4) - 0.5
	mesh_points= mesh_points*np.expand_dims(dimension,0)
	# mesh_points = (mesh_points + 0.5) / voxel_size[1] - 0.5
	# mesh_points = (mesh_points + 0.5) / voxel_size[2] - 0.5
	if not os.path.exists(os.path.dirname(mesh_filename)):
		os.makedirs(os.path.dirname(mesh_filename))
	

	

	# utils.save_obj_data(f"{mesh_filename}.obj", mesh_points, faces)
	mesh= pymesh.form_mesh(np.array(mesh_points), np.array(faces))
	# mesh = extract_largest_component(mesh)
	pymesh.save_mesh(f"{mesh_filename}.obj", mesh, ascii=True)
	mesh = o3d.io.read_triangle_mesh(f"{mesh_filename}.obj")
	smooth_mesh= mesh.filter_smooth_simple(number_of_iterations=10)
	o3d.io.write_triangle_mesh(f"{mesh_filename}_smooth.obj", smooth_mesh)

def create_CAD_mesh(generator, shape_code, shape_3d, CAD_mesh_filepath):
    """
    Reconstruct shapes with sketch-extrude operations.
    
    Notes:
        - This function currently contains no implementation and serves as a stub.
    """
    pass


def draw_2d_im_sketch(shape_code, generator, sk_filepath):
    """
    Draw a 2D sketch.
    
    Notes:
        - This function currently contains no implementation and serves as a stub.
    """
    pass