import torch
import torch.nn as nn
import torch.nn.functional as F





def CSG(current, xyz, drill_ind, points_gt):
    cylinders = Cylinder(points_gt,drill_ind,xyz) # (b,N)
    carved = torch.max(torch.cat([current.unsqueeze(-1),-cylinders.unsqueeze(-1)],-1),-1)[0] # (b,N)
    return carved,cylinders
    
    
    
def Cylinder(points,radius,centers):
    '''
    points:  (b,N,3)
    ind:     (b)
    centers: (b,100,3)
    '''
  
    xy,z = centers

    r = radius.unsqueeze(-1)
    cylinders = torch.sum((points.unsqueeze(1)[:,:,:,:2]-xy.unsqueeze(2))**2,-1)-r**2 # (b,100,N)
    cylinders = torch.min(cylinders,1)[0]# (b,N)
    cylinders_z = torch.cat([cylinders.unsqueeze(-1),(z-points[:,:,2]).unsqueeze(-1)],-1) # (b,N,2)
    cylinders_z = torch.max(cylinders_z,-1)[0] # (b,N)

    return  cylinders_z# (b,N)





    
def Cube(points,params):
    '''
    points:  (b,N,3)
    params:     (b,3)
    origin:     (b,3)
    '''
    params = params.unsqueeze(1) # (b,1,4)

    cube = torch.max(torch.abs((points)/(params[:,:,:3])*2),-1)[0]-1
    
    return torch.tanh(cube)







