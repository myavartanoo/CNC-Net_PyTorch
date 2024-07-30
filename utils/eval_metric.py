#from lfd import LightFieldDistance
import trimesh
import torch
import numpy as np
from pytorch3d.ops.knn import knn_gather, knn_points



def get_chamfer_distance(pred_points, gt_points):
    """
    pred_points (B, N, C)
    gt_points (B, N, C)
    """

    batch_CD = []
    for i in range(pred_points.shape[0]):
        pred_point = pred_points[i]
        gt_point = gt_points[i]
        gt_num_points = gt_point.shape[0]
        pred_num_points = pred_point.shape[0]

        points_gt_matrix = gt_point.unsqueeze(1).expand(
            [gt_point.shape[0], pred_num_points, gt_point.shape[-1]]
        )
        points_pred_matrix = pred_point.unsqueeze(0).expand(
            [gt_num_points, pred_point.shape[0], pred_point.shape[-1]]
        )

        distances = (points_gt_matrix - points_pred_matrix).pow(2).sum(dim=-1)
        match_pred_gt = distances.argmin(dim=0)
        match_gt_pred = distances.argmin(dim=1)

        dist_pred_gt = (pred_point - gt_point[match_pred_gt]).pow(2).sum(dim=-1).mean()
        dist_gt_pred = (gt_point - pred_point[match_gt_pred]).pow(2).sum(dim=-1).mean()


        chamfer_distance = dist_pred_gt + dist_gt_pred
        batch_CD.append(chamfer_distance)


    return np.mean(batch_CD)


def IOU(current, inds_inout):
    in_current = (current[0]<0).nonzero().squeeze().cpu().numpy()
    out_current = (current[0]>0).nonzero().squeeze().cpu().numpy()

    in_gt =(inds_inout[0]<0).nonzero().squeeze().cpu().numpy()
    out_gt = (inds_inout[0]>0).nonzero().squeeze().cpu().numpy()

    in_intersection = np.intersect1d(in_current, in_gt)
    in_union = np.union1d(in_current, in_gt)
    #out_intersection = np.intersect1d(out_current, out_gt)
    #out_union = np.union1d(out_current, out_gt)

    #return (in_intersection.shape[0]+out_intersection.shape[0])/(in_union.shape[0]+out_union.shape[0])
    if in_union.shape[0]!=0:
       return in_intersection.shape[0]/in_union.shape[0]
    else:
       return in_union.shape[0]


def LFD(mesh1_v, mesh1_face, mesh2_v, mesh2_face):
    value = LightFieldDistance(verbose = True).get_distance(mesh1_v, mesh1_face, mesh2_v, mesh2_face)

    return value

if __name__ == '__main__':
    # example for LFD
    #mesh1 = trimesh.load('/mnt/disk1/mchiash2/abc_all/meshes/00030572.off')
    #mesh2 = trimesh.load('/mnt/disk1/mchiash2/abc_all/meshes/00030517.off')
    #lfd = LFD(mesh1.vertices, mesh1.faces, mesh2.vertices, mesh2.faces)

    a1 = 2*torch.rand((1,1000 ))-1
    b1 = 2*torch.rand((1,1000))-1
    print(IOU(a1, b1))
    print(custom_iou_loss(a1, b1))

