import torch
from pytorch3d.ops import knn_points


def pairwise_distances(x, y):
    # Computes pairwise distances between each point in x and y
    # x: [N, D]
    # y: [M, D]
    # Output: [N, M]
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    dist = torch.norm(diff, p=2, dim=-1)
    return dist

def filter_points2(point_cloud, radius_value, num_threshold=3):
    # Compute pairwise distances
    point_cloud = point_cloud[0]

    distances = pairwise_distances(point_cloud, point_cloud)
    
    # Count neighbors within the radius
    neighbor_counts = (distances < radius_value).sum(dim=1) - 1  # subtract 1 to exclude the point itself
    
    # Filter out points with less neighbors than the threshold
    mask = neighbor_counts > num_threshold
    filtered_point_cloud = point_cloud[mask]
    
    return filtered_point_cloud.unsqueeze(0)


def filter_points(point_cloud, dimension):
    nn_dists, nn_idx, nn = knn_points(point_cloud*torch.reciprocal(dimension), point_cloud*torch.reciprocal(dimension), K=7)
    nn_dists = torch.sqrt(nn_dists)
    threshold = torch.min(nn_dists[0,:,1:])+0.0005
    pcd_filtered = point_cloud[:, nn_dists[0,:,1:].mean(1) < threshold]

    
    return pcd_filtered
if __name__ == "__main__":
    point_cloud = torch.randn((1, 1000,3))
    radius_value = 0.5
    filtered_points = filter_points2(point_cloud)
    import pdb; pdb.set_trace()