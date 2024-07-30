import open3d as o3d
import trimesh
import numpy as np
import pymesh       
import trimesh
import torch
import pytorch3d
from tqdm import tqdm
from utils.output_xyz import output_xyz
from pytorch3d.transforms.transform3d import Transform3d


def intersect(p1, p2, p3, p4):
   # Calculate differences
   d1 = p2 - p1
   d2 = p4 - p3

   # Calculate cross product
   cross_product = np.cross(d1, d2)

   # Check if lines are parallel
   if cross_product == 0:
      # Handle parallel lines
      return False

   # Calculate parameters t and u
   t = np.cross(p3 - p1, d2) / cross_product
   u = np.cross(p3 - p1, d1) / cross_product

   # Check if intersection point lies on both line segments
   if 0 <= t <= 1 and 0 <= u <= 1:
      return True

   return False


                              

def extract_largest_component(mesh):
    # Separate the mesh into connected components
    components = pymesh.separate_mesh(mesh)

    # Find the largest component based on the number of vertices
    largest_mesh = max(components, key=lambda component: component.num_vertices)

    return largest_mesh




def visualize(cnc_parameters, all_points, out_dir):
   """
   drill_xyz (B, 50, 100, 3)
   drill_radius (B, 50, 1)
   R_M (B, N, 3)
   out_dir
   """   
   def get_mesh(trajectory, radius):
      vertices = []

      tangents = trajectory[1:]-trajectory[:-1]
      tangents = np.vstack([tangents, tangents[-1]])

      tangents_normalized = tangents/np.linalg.norm(tangents, axis=1)[:, np.newaxis]

      normals = np.cross(tangents_normalized, np.array([0., 0., 1.]))
      normals_normalized = normals/  np.linalg.norm(normals, axis=1)[:, np.newaxis]




      points_left = trajectory + radius * normals_normalized
      points_right = trajectory - radius*normals_normalized
      vertices = np.vstack([points_left, points_right]).reshape(-1, 3) 

      faces = []
      for j in range(99):
         p1 = j*2
         p2 = p1+1
         p3 = p1 +2
         p4 = p1 +3

         faces.append(np.array([p1, p2, p3]))
         faces.append(np.array([p2, p4, p3]))

      return vertices, faces


   def extrude_mesh(vertices, faces):
      new_vertices = []
      new_faces = []
      num_original_vertices = len(vertices)
      for vertex in vertices:
         x,y,z = vertex
         new_vertices.append(np.array([x,y,z]))
         new_vertices.append(np.array([x,y,1]))



      for i in range((num_original_vertices//2)):
         bottom_left = 2*i 
         bottom_right = 2*i+(num_original_vertices)
         top_left = 2*i+1
         top_right = 2*i+(num_original_vertices)+1
         prev_bottom_left = 2*i-2 
         prev_bottom_right = 2*i+(num_original_vertices)-2
         prev_top_left = 2*i+1-2
         prev_top_right = 2*i+(num_original_vertices)+1-2       


         if i==0:
            new_faces.append(np.array([bottom_left, top_left, bottom_right]))
            new_faces.append(np.array([bottom_right, top_left, top_right]))

         else:
            new_faces.append(np.array([bottom_left, prev_top_left, prev_bottom_left]))
            new_faces.append(np.array([bottom_left, top_left, prev_top_left]))
            new_faces.append(np.array([bottom_right, prev_bottom_right, prev_top_right]))
            new_faces.append(np.array([bottom_right, prev_top_right, top_right]))


            new_faces.append(np.array([top_left, prev_top_right, prev_top_left]))  
            new_faces.append(np.array([top_left, top_right, prev_top_right]))  
            new_faces.append(np.array([bottom_left, prev_bottom_left, prev_bottom_right]))  
            new_faces.append(np.array([bottom_left, prev_bottom_right, bottom_right]))    


         if i==((num_original_vertices//2)-1):
            new_faces.append(np.array([bottom_left, bottom_right, top_left]))
            new_faces.append(np.array([bottom_right, top_right, top_left]))            

         
      return new_vertices, new_faces

   index = torch.arange(0, 100, 10)
   mill_xy = cnc_parameters['mill_xy']
   # mill_xy = mill_xy[:,index]
   mill_z = cnc_parameters['mill_z']
   mill_rot_param = cnc_parameters['mill_rot_param']
   mill_radius = cnc_parameters['mill_radius']

   # print(mill_xy.shape)



   drill_xy = cnc_parameters['drill_xy']
   drill_z = cnc_parameters['drill_z']
   drill_rot_param = cnc_parameters['drill_rot_param']
   drill_radius = cnc_parameters['drill_radius']

   mill_xyz = torch.cat((mill_xy, mill_z.unsqueeze(-1).repeat(1, mill_xy.shape[1], 1)),2)
   

   if drill_xy.shape[0]!=0:
      drill_xyz = torch.cat((drill_xy, drill_z.unsqueeze(-1)), 2)
      drill_xyz = drill_xyz.detach().cpu()
   
   mill_xyz = mill_xyz.detach().cpu().numpy()
   mill_radius = mill_radius.detach().cpu().numpy()
   mill_rot_param = mill_rot_param.detach().cpu()
   drill_xy = drill_xy.detach().cpu().numpy()
   drill_z = drill_z.detach().cpu().numpy()
   drill_rot_param = drill_rot_param.detach().cpu()
   drill_radius = drill_radius.detach().cpu().numpy()
   all_points = all_points.detach().cpu().numpy()

   num_iteration = mill_xyz.shape[0]
   points_in_path = mill_xyz.shape[1]


   


   final_shapes = []
   extruded_meshes = []
   current = pymesh.generate_box_mesh(box_min=np.min(all_points[0], 0),box_max= np.max(all_points[0], 0))
   # pymesh.save_mesh(out_dir+'initial_box.ply', current, ascii=True)



   for i in range(drill_xy.shape[0]):
      R_M = pytorch3d.transforms.axis_angle_to_matrix(drill_rot_param[i])
      rot = Transform3d().rotate(R_M)      
      R_M_inv = torch.from_numpy(np.linalg.inv(R_M.numpy()))
      rot_inv = Transform3d().rotate(R_M_inv)
      current = pymesh.form_mesh((rot.transform_points(torch.Tensor(current.vertices))).numpy(), np.array(current.faces))
      r = drill_radius[i][0]
      cylinder_drill = pymesh.generate_cylinder(drill_xyz[i,0],torch.Tensor([0,0,1]), r, r, num_segments=16)
      
      current = pymesh.boolean(current, cylinder_drill, operation="difference", engine ="cork")
      current = pymesh.form_mesh((rot_inv.transform_points(torch.Tensor(current.vertices))).numpy(), np.array(current.faces))
      





   current_meshes = []
   for j in tqdm(range(num_iteration)):
      

      trajectory = mill_xyz[j]
      radius = mill_radius[j,0].item()

      vertices,faces = get_mesh(trajectory, radius)
      mesh_2d = pymesh.form_mesh(np.array(vertices), np.array(faces))

      R_M = pytorch3d.transforms.axis_angle_to_matrix(mill_rot_param[j])
      rot = Transform3d().rotate(R_M)
      R_M_inv = torch.from_numpy(np.linalg.inv(R_M.numpy()))
      rot_inv = Transform3d().rotate(R_M_inv)



      current = pymesh.form_mesh((rot.transform_points(torch.Tensor(current.vertices))).numpy(), np.array(current.faces))



      new_vertices, new_faces = extrude_mesh(vertices, faces)



      extruded_mesh = pymesh.form_mesh(np.array(new_vertices), np.array(new_faces))
      extruded_mesh = pymesh.collapse_short_edges(extruded_mesh, rel_threshold=0.05)[0]
      # pymesh.save_mesh(out_dir+'extruded_mesh_'+str(j)+'.ply', extruded_mesh, ascii=True)



      r = mill_radius[j][0]


      cylinder_s = pymesh.generate_cylinder(mill_xyz[j,0].astype(np.float64),  np.asarray([0,0,1]), r, r, num_segments=16)   
      cylinder_e = pymesh.generate_cylinder(mill_xyz[j,-1].astype(np.float64), np.asarray([0,0,1]), r, r, num_segments=16)  


      


      current = pymesh.boolean(current, extruded_mesh, operation="difference", engine ="cork")
      # pymesh.save_mesh(out_dir+'current_shape_'+str(j)+'.ply', current, ascii=True)
      current = pymesh.collapse_short_edges(current, rel_threshold=0.05)[0]

      

      current = pymesh.boolean(current, cylinder_s, operation="difference", engine ="cork")
      current = pymesh.boolean(current, cylinder_e, operation="difference", engine ="cork")
      current = pymesh.collapse_short_edges(current, rel_threshold=0.05)[0]


      
      current = pymesh.form_mesh((rot_inv.transform_points(torch.Tensor(current.vertices))).numpy(), np.array(current.faces))

   current = extract_largest_component(current)
   pymesh.save_mesh(out_dir+'.ply', current, ascii=True)
         






def visualize_old(cnc_parameters, all_points, out_dir):
   """
   drill_xyz (B, 50, 100, 3)
   drill_radius (B, 50, 1)
   R_M (B, N, 3)
   out_dir
   """   
   def get_mesh(trajectory, radius):
      vertices = []
      tangents = trajectory[1:]-trajectory[:-1]
      tangents = np.vstack([tangents, tangents[-1]])
      tangents_normalized = tangents/np.linalg.norm(tangents, axis=1)[:, np.newaxis]

      normals = np.cross(tangents_normalized, np.array([0., 0., 1.]))
      normals_normalized = normals/  np.linalg.norm(normals, axis=1)[:, np.newaxis]

      points_left = trajectory + radius * normals_normalized
      points_right = trajectory - radius*normals_normalized
      vertices = np.vstack([points_left, points_right]).reshape(-1, 3) 

      faces = []
      for j in range(99):
         p1 = j*2
         p2 = p1+1
         p3 = p1 +2
         p4 = p1 +3

         faces.append(np.array([p1, p2, p3]))
         faces.append(np.array([p2, p4, p3]))

      return vertices, faces


   def extrude_mesh(vertices, faces):
      new_vertices = []
      new_faces = []
      num_original_vertices = len(vertices)
      for vertex in vertices:
         x,y,z = vertex
         new_vertices.append(np.array([x,y,z]))
         new_vertices.append(np.array([x,y,z+1]))



      for i in range((num_original_vertices//2)):
         bottom_left = 2*i 
         bottom_right = 2*i+(num_original_vertices)
         top_left = 2*i+1
         top_right = 2*i+(num_original_vertices)+1
         prev_bottom_left = 2*i-2 
         prev_bottom_right = 2*i+(num_original_vertices)-2
         prev_top_left = 2*i+1-2
         prev_top_right = 2*i+(num_original_vertices)+1-2        
         if i==0:
            new_faces.append(np.array([bottom_left, bottom_right, top_left]))
            new_faces.append(np.array([bottom_right, top_right, top_left]))

         else:
            new_faces.append(np.array([bottom_left, prev_bottom_left, prev_top_left]))
            new_faces.append(np.array([bottom_left, top_left, prev_top_left]))
            new_faces.append(np.array([bottom_right, prev_bottom_right, prev_top_right]))
            new_faces.append(np.array([bottom_right, top_right, prev_top_right]))
            new_faces.append(np.array([top_left, prev_top_right, prev_top_left]))  
            new_faces.append(np.array([top_left, top_right, prev_top_right]))  
            new_faces.append(np.array([bottom_left, prev_bottom_right, prev_bottom_left]))  
            new_faces.append(np.array([bottom_left, bottom_right, prev_bottom_right]))              

         if i==((num_original_vertices//2)-1):
            new_faces.append(np.array([bottom_left, bottom_right, top_left]))
            new_faces.append(np.array([bottom_right, top_right, top_left]))            

         
      return new_vertices, new_faces

   mill_xy = cnc_parameters['mill_xy']
   mill_z = cnc_parameters['mill_z']
   mill_rot_param = cnc_parameters['mill_rot_param']
   mill_radius = cnc_parameters['mill_radius']



   drill_xy = cnc_parameters['drill_xy']
   drill_z = cnc_parameters['drill_z']
   drill_rot_param = cnc_parameters['drill_rot_param']
   drill_radius = cnc_parameters['drill_radius']

   mill_xyz = torch.cat((mill_xy, mill_z.unsqueeze(-1).repeat(1, mill_xy.shape[1], 1)),2)
   
   
   #drill_xyz = torch.cat((drill_xy, drill_z.unsqueeze(-1)), 2)

   
   mill_xyz = mill_xyz.detach().cpu()
   mill_radius = mill_radius.detach().cpu().numpy()
   mill_rot_param = mill_rot_param.detach().cpu()
   drill_xy = drill_xy.detach().cpu().numpy()
   drill_z = drill_z.detach().cpu().numpy()
   drill_rot_param = drill_rot_param.detach().cpu()
   drill_radius = drill_radius.detach().cpu().numpy()
   #drill_xyz = drill_xyz.detach().cpu()
   all_points = all_points.detach().cpu().numpy()

   num_iteration = mill_xyz.shape[0]
   points_in_path = mill_xyz.shape[1]


   


   final_shapes = []
   extruded_meshes = []
   initial = pymesh.generate_box_mesh(box_min=np.min(all_points[0], 0)-0.015 ,box_max= np.max(all_points[0], 0)+0.015)
   pymesh.save_mesh(out_dir+'initial_box.ply', initial, ascii=True)



   for i in range(drill_xy.shape[0]):
      R_M_inv = pytorch3d.transforms.axis_angle_to_matrix(-drill_rot_param[i])
      rot_inv = Transform3d().rotate(R_M_inv)
      r = drill_radius[i][0]
      cylinder_drill = pymesh.generate_cylinder(rot_inv.transform_points(drill_xyz[i,0].unsqueeze(0))[0], rot_inv.transform_points((drill_xyz[i,0]+torch.Tensor([0,0,1])).unsqueeze(0))[0], r, r, num_segments=16)
      
      current = pymesh.boolean(current, cylinder_drill, operation="difference", engine ="cork")
   





   diffs = []
   for j in tqdm(range(num_iteration)):
      

      trajectory = mill_xyz[j]
      radius = mill_radius[j,0].item()

      vertices,faces = get_mesh(trajectory, radius)
      mesh_2d = pymesh.form_mesh(np.array(vertices), np.array(faces))

      R_M = pytorch3d.transforms.axis_angle_to_matrix(mill_rot_param[j])
      rot = Transform3d().rotate(R_M)
      R_M_inv = pytorch3d.transforms.axis_angle_to_matrix(-mill_rot_param[j])
      rot_inv = Transform3d().rotate(R_M_inv)


      initial = pymesh.form_mesh((rot.transform_points(torch.Tensor(initial.vertices))).numpy(), np.array(initial.faces))



      new_vertices, new_faces = extrude_mesh(vertices, faces)



      extruded_mesh = pymesh.form_mesh(np.array(new_vertices), np.array(new_faces))
      extruded_mesh = pymesh.collapse_short_edges(extruded_mesh, rel_threshold=0.05)[0]
      pymesh.save_mesh(out_dir+'extruded_mesh_'+str(j)+'.ply', extruded_mesh, ascii=True)





      r = mill_radius[j][0]

      cylinder_s = pymesh.generate_cylinder(mill_xyz[j,0],  mill_xyz[j,0]+torch.Tensor([0,0,1]), r, r, num_segments=16)   
      cylinder_e = pymesh.generate_cylinder(mill_xyz[j,-1], mill_xyz[j,-1]+torch.Tensor([0,0,1]), r, r, num_segments=16)  


      


      diff = pymesh.boolean(initial, extruded_mesh, operation="difference", engine ="cork")
      diff = pymesh.collapse_short_edges(diff, rel_threshold=0.05)[0]

      

      diff = pymesh.boolean(diff, cylinder_s, operation="difference", engine ="cork")
      diff = pymesh.boolean(diff, cylinder_e, operation="difference", engine ="cork")
      diff = pymesh.collapse_short_edges(diff, rel_threshold=0.05)[0]


      
      diff = pymesh.form_mesh((rot_inv.transform_points(torch.Tensor(diff.vertices))).numpy(), np.array(diff.faces))
      pymesh.save_mesh(out_dir+'diff_'+str(j)+'.ply', diff, ascii=True)
      diffs.append(diff)

      


   current = diffs[0]
   for diff in diffs[1:]:
      current = pymesh.boolean(current, diff, operation="intersection", engine ="cork")

   current = extract_largest_component(current)
   pymesh.save_mesh(out_dir+'_final_shape.ply', current, ascii=True)




   import pdb; pdb.set_trace()
