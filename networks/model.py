import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pytorch3d
import copy

from pytorch3d.ops.knn import knn_gather, knn_points
from networks.primitives import CSG
from pytorch3d.transforms.transform3d import Transform3d
from networks.capri import Encoder
from utils.denoise_points import filter_points


def chamfer_distance_kdtree(x, y):
    ''' KD-tree based implementation of the Chamfer distance.
    Args:
        points1 (batch, 3, num_on_points)
        points2 (batch, 3, num_on_points)
    '''


    #x = torch.transpose(x,1,2)
    #y = torch.transpose(y,1,2)



    x_nn = knn_points(x, y, K=1)
    y_nn = knn_points(y, x, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)



    # Apply point reduction
    cham_x = torch.sqrt(cham_x)  # (N,)
    cham_y = torch.sqrt(cham_y)  # (N,)
    


    return cham_x,cham_y
    
mse = nn.MSELoss(reduction = 'mean')



class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class Model(nn.Module):
    def _initialize_weights(self):
       
        for m in self.f_rot:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity = 'relu')
                if m.bias is not None:
                    init.constant_(m.bias,0)

        for m in self.f_radius:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity = 'relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)    

        for m in self.p_xy:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity = 'relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0) 
                    
        for m in self.p_z:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity = 'relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)                                                                 


                            
    def __init__(self, ef_dim=256):
        super(Model, self).__init__()
        self.ef_dim = ef_dim    
        self.lstm = nn.LSTM(input_size=self.ef_dim+3, hidden_size=self.ef_dim, num_layers=2, batch_first=True) #input_size = 2048

        f_rot = []
        f_rot.append(nn.Linear(self.ef_dim, self.ef_dim, bias=True))
        f_rot.append(nn.ReLU(inplace=True))
        f_rot.append(ResnetBlockFC(self.ef_dim))
        f_rot.append(ResnetBlockFC(self.ef_dim))
        f_rot.append(ResnetBlockFC(self.ef_dim))
        f_rot.append(ResnetBlockFC(self.ef_dim))
        f_rot.append(ResnetBlockFC(self.ef_dim))
        f_rot.append(nn.Linear(self.ef_dim , 2, bias=True))
        self.f_rot = nn.Sequential(*f_rot)

        f_radius = []
        f_radius.append(nn.Linear(self.ef_dim, self.ef_dim, bias=True))
        f_radius.append(nn.ReLU(inplace=True))
        f_radius.append(ResnetBlockFC(self.ef_dim))
        f_radius.append(ResnetBlockFC(self.ef_dim))
        f_radius.append(ResnetBlockFC(self.ef_dim))
        f_radius.append(ResnetBlockFC(self.ef_dim))
        f_radius.append(ResnetBlockFC(self.ef_dim))
        f_radius.append(nn.Linear(self.ef_dim , 4, bias=True))
        self.f_radius = nn.Sequential(*f_radius)
        


        p_xy = []
        p_xy.append(nn.Linear(self.ef_dim+1, self.ef_dim , bias=True))
        p_xy.append(nn.ReLU(inplace=True))
        p_xy.append(ResnetBlockFC(self.ef_dim))
        p_xy.append(ResnetBlockFC(self.ef_dim))
        p_xy.append(ResnetBlockFC(self.ef_dim))
        p_xy.append(ResnetBlockFC(self.ef_dim))
        p_xy.append(ResnetBlockFC(self.ef_dim))       
        p_xy.append(nn.Linear(self.ef_dim , 2, bias=True))	
        self.p_xy = nn.Sequential(*p_xy)

        p_z = []
        p_z.append(nn.Linear(self.ef_dim, self.ef_dim, bias=True))
        p_z.append(nn.ReLU(inplace=True))
        p_z.append(ResnetBlockFC(self.ef_dim))
        p_z.append(ResnetBlockFC(self.ef_dim))
        p_z.append(ResnetBlockFC(self.ef_dim))
        p_z.append(ResnetBlockFC(self.ef_dim))
        p_z.append(ResnetBlockFC(self.ef_dim))
        p_z.append(nn.Linear(self.ef_dim , 1, bias=True))
        self.p_z = nn.Sequential(*p_z)

        f_rot_drill = []
        f_rot_drill.append(nn.Linear(self.ef_dim, self.ef_dim, bias=True))
        f_rot_drill.append(nn.ReLU(inplace=True))
        f_rot_drill.append(ResnetBlockFC(self.ef_dim))
        f_rot_drill.append(ResnetBlockFC(self.ef_dim))
        f_rot_drill.append(ResnetBlockFC(self.ef_dim))
        f_rot_drill.append(ResnetBlockFC(self.ef_dim))
        f_rot_drill.append(ResnetBlockFC(self.ef_dim))
        f_rot_drill.append(nn.Linear(self.ef_dim , 2, bias=True))
        self.f_rot_drill = nn.Sequential(*f_rot_drill)



        p_xyz_drill = []
        p_xyz_drill.append(nn.Linear(self.ef_dim, self.ef_dim , bias=True))
        p_xyz_drill.append(nn.ReLU(inplace=True))
        p_xyz_drill.append(ResnetBlockFC(self.ef_dim))
        p_xyz_drill.append(ResnetBlockFC(self.ef_dim))
        p_xyz_drill.append(ResnetBlockFC(self.ef_dim))
        p_xyz_drill.append(ResnetBlockFC(self.ef_dim))
        p_xyz_drill.append(ResnetBlockFC(self.ef_dim))       
        p_xyz_drill.append(nn.Linear(self.ef_dim , 3, bias=True))	
        self.p_xyz_drill = nn.Sequential(*p_xyz_drill)



        f_radius_drill = []
        f_radius_drill.append(nn.Linear(self.ef_dim, self.ef_dim, bias=True))
        f_radius_drill.append(nn.ReLU(inplace=True))
        f_radius_drill.append(ResnetBlockFC(self.ef_dim))
        f_radius_drill.append(ResnetBlockFC(self.ef_dim))
        f_radius_drill.append(ResnetBlockFC(self.ef_dim))
        f_radius_drill.append(ResnetBlockFC(self.ef_dim))
        f_radius_drill.append(ResnetBlockFC(self.ef_dim))
        f_radius_drill.append(nn.Linear(self.ef_dim , 4, bias=True))
        self.f_radius_drill = nn.Sequential(*f_radius_drill)



                
        self.softmax = nn.Softmax(dim=-1)
        
                
        self.Encoder = Encoder()



    def forward(self,  initial_current, initial_current_high, all_points, all_points_high, inds_inout,   n_step,out_dir, best_iou, dimension, epoch, test):
            
        w = 1000

        hidden = None # Initialize the hidden state

        Loss_occ = []
        current = initial_current
        current_high = initial_current_high
        loss_occ = 0
        loss_ends = 0
        loss_occ_prev = -1
        loss_occ_next = 0
        it_mill = 0
        loss_roi = 0
        Loss_occ_drill = 0
        vox_size = 64      


        mill_xy = []
        mill_z = []
        mill_radius = []
        mill_rot_param = []
        
        # Milling operations
        while np.abs(loss_occ_next-loss_occ_prev)>0.0001  and it_mill <20:
            if it_mill>0:
               loss_occ_prev = loss_occ.detach().cpu().numpy()
 
 
            current_roi = torch.minimum(inds_inout, torch.sign(-current).detach())
 
            current_feat = torch.cat([self.Encoder(current_roi.view(1,1,vox_size,vox_size,vox_size)),dimension.float()],-1)
            

            itt = torch.zeros_like(current_feat[:,:1])
            itt[:, 0] = it_mill    

            lstm_output, hidden = self.lstm(current_feat, hidden)
            lstm_output = lstm_output.squeeze(1)                                 
        
            N = n_step
            steps = torch.arange(start=0, end=1, step=1/N, device=lstm_output.device)
            steps = steps.unsqueeze(0).expand(lstm_output.shape[0], -1).unsqueeze(-1)
            steps_high = torch.arange(start=0, end=1, step=1/(5*N), device=lstm_output.device)
            steps_high = steps_high.unsqueeze(0).expand(lstm_output.shape[0], -1).unsqueeze(-1)


            rot_param = torch.tanh(self.f_rot(lstm_output))*np.pi
            angles = torch.cat([rot_param,torch.zeros_like(rot_param[...,:1])],-1)
            
            
            radius = self.f_radius(lstm_output)
            tool_options = torch.Tensor([0.025, 0.05, 0.075, 0.1]).cuda()
            
            tool_distribution = F.gumbel_softmax(radius, tau=1, hard=True)
            tool_radius = torch.sum(tool_distribution*tool_options, dim=-1, keepdim=True)
            mill_radius.append(tool_radius)


            if it_mill>0:
               xy_prev = xy
            xy = self.p_xy(torch.cat([lstm_output.unsqueeze(1).expand(-1,N,-1),steps],-1))
            xy_high = self.p_xy(torch.cat([lstm_output.unsqueeze(1).expand(-1,5*N,-1),steps_high],-1))
            z = self.p_z(lstm_output) 
            mill_xy.append(xy)
            mill_z.append(z)
            path_xyz = [xy, z]



            R_M = pytorch3d.transforms.axis_angle_to_matrix(angles)
            rot = Transform3d().cuda().rotate(R_M)

            mill_rot_param.append(angles)
            
            current,cyl = CSG(current, path_xyz, tool_radius, rot.transform_points(all_points))
            if test:
            	current_high,cyl_high = CSG(current_high, path_xyz, tool_radius, rot.transform_points(all_points_high))


            loss_occ = mse(torch.sign(current), inds_inout)
            

            ch1,ch2 = chamfer_distance_kdtree(torch.cat([path_xyz[0],path_xyz[1].unsqueeze(1).expand(-1,path_xyz[0].shape[1],-1)],-1)[:,:,:2], rot.transform_points(all_points[:,current_roi[0]>0])[:,:,:2]) 
            loss_ends += (torch.mean(ch1) + torch.mean(ch2))


            loss_roi += mse(torch.tanh(w*cyl[inds_inout<0]), torch.ones_like(cyl[inds_inout<0]))

            


            loss_occ_next = loss_occ.detach().cpu().numpy()

            it_mill += 1

        

        it_path = it_mill-1
        loss_occ_prev = loss_occ_next-1

        mill_xy = torch.stack(mill_xy, dim=1).squeeze(0)
        mill_z = torch.stack(mill_z, dim=1).squeeze(0)
        mill_rot_param = torch.stack(mill_rot_param, dim =1).squeeze(0)
        mill_radius = torch.stack(mill_radius, dim=1).squeeze(0)

        drill_xy = []
        drill_z = []
        drill_rot_param = []
        drill_radius = []

        Loss_occ_drill = mse(torch.tanh(w*current), inds_inout) 
        
        current = copy.deepcopy(current.detach())
        loss_nroi = 0
        it_drill = 0
        
        
        
        
        
        
        # Drilling operations
        while np.abs(loss_occ_next-loss_occ_prev)>0.1 and it_drill <20 and best_iou>0.92: #and epoch>50:
        

            loss_occ_prev = loss_occ.detach().cpu().numpy()
            current_roi = torch.minimum(inds_inout, torch.sign(-current).detach())


            current_feat = torch.cat([self.Encoder(current_roi.view(1,1,vox_size,vox_size,vox_size)),dimension.float()],-1)

            itt = torch.zeros_like(current_feat[:,:1])
            itt[:, 0] = it_drill    


            lstm_output, hidden = self.lstm(current_feat, hidden)
            lstm_output = lstm_output.squeeze(1)                                 
        
            N = 1
            steps = torch.arange(start=0, end=1, step=1/N, device=lstm_output.device)
            steps = steps.unsqueeze(0).expand(lstm_output.shape[0], -1).unsqueeze(-1)
 
            rot_param = torch.tanh(self.f_rot_drill(lstm_output))*np.pi 
            angles = torch.cat([rot_param,torch.zeros_like(rot_param[...,:1])],-1)

            radius = self.f_radius_drill(lstm_output)
            tool_options = torch.Tensor([0.01, 0.02, 0.03, 0.04]).cuda()

            tool_distribution = F.gumbel_softmax(radius, tau=1, hard=True)
            tool_radius = torch.sum(tool_distribution*tool_options, dim=-1, keepdim=True)
            
            drill_radius.append(tool_radius)


            xyz = self.p_xyz_drill(lstm_output).unsqueeze(1)


            xy = xyz[:,:,0:2]
            z = xyz[:,:,-1]

            drill_xy.append(xy)
            drill_z.append(z) 

            path_xyz = [xy, z]
               

            R_M = pytorch3d.transforms.axis_angle_to_matrix(angles)
            rot = Transform3d().cuda().rotate(R_M)
                     
            drill_rot_param.append(angles)



            current,cyl = CSG(current, path_xyz, tool_radius, rot.transform_points(all_points))
            if test:            
            	current_high,cyl_high = CSG(current_high, path_xyz, tool_radius, rot.transform_points(all_points_high))


            loss_occ = mse(torch.sign(current), inds_inout)
            #Loss_occ.append(loss_occ)

            filtered_points = filter_points(all_points[:,current_roi[0]>0], dimension)
            points_remaining = filtered_points.shape[1]

            ch1,ch2 = chamfer_distance_kdtree(torch.cat([path_xyz[0],path_xyz[1].unsqueeze(1).expand(-1,path_xyz[0].shape[1],-1)],-1)[:,:,:2], (rot.transform_points(filtered_points)[:,:,:2])) # Should be checked
            loss_ends += (torch.mean(ch1) + torch.mean(ch2)) 
            

            mask = torch.ones_like(cyl)
            mask[current_roi>0] = -1
            loss_nroi += mse(torch.tanh(w*cyl), mask) 



            loss_roi += mse(torch.tanh(w*cyl[inds_inout<0]), torch.ones_like(cyl[inds_inout<0]))



            loss_occ_next = loss_occ.detach().cpu().numpy()
            
            it_drill += 1
       



        if len(drill_xy)!=0:
            drill_xy = torch.stack(drill_xy, dim=1).squeeze(0)
            drill_z = torch.stack(drill_z, dim=1).squeeze(0)
            drill_rot_param = torch.stack(drill_rot_param, dim =1).squeeze(0)
            drill_radius = torch.stack(drill_radius, dim=1).squeeze(0)      

        else:
            drill_current = drill_xy = drill_z = drill_rot_param = drill_radius = torch.Tensor([])        
         

          
 
         

        loss_total = Loss_occ_drill + loss_ends/(it_mill+it_drill+2) + loss_roi/(it_mill+it_drill+2) + loss_nroi/(it_drill+1)
        if test:        
        	return loss_total, current, current_high
        else:        
        	return loss_total, current        	
