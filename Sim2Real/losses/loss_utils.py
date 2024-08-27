import torch


def closest_distance_with_batch(p1, p2, is_sum=True):
    """
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param is_sum: whehter to return the summed scalar or the separate distances with indices
    :return: the distances from p1 to the closest points in p2
    """
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    min_dist, min_indice = torch.min(dist, dim=2)
    dist_scalar = torch.sum(min_dist)

    if is_sum:
        return dist_scalar
    else:
        return min_dist, min_indice


def point2sphere_distance_with_batch(p1, p2):
    """
    Calculate the distances from points in p1 to the closest spheres in p2.

    :param p1: size[B, N, 3] - Batch of 3D points
    :param p2: size[B, M, 4] - Batch of spheres represented as (x, y, z, radius)
    :return: p2p_min_dist: size[B, N] - Minimum distances from p1 to the closest center points in p2
             p2s_min_dist: size[B, N] - Minimum distances from p1 to the closest sphere surfaces in p2
    """
    assert p1.size(0) == p2.size(0) and p1.size(2) == 3 and p2.size(2) == 4

    # Expand dimensions for broadcasting
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    # Repeat tensors to have compatible shapes
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    # Separate sphere representations into positions and radii
    p2_xyzr = p2
    p2 = p2_xyzr[:, :, :, 0:3]
    p2_r = p2_xyzr[:, :, :, 3]              # B N M

    # Calculate the vector from p1 points to p2 spheres
    dist = torch.add(p1, torch.neg(p2))     # B N M 3
    # Calculate the Euclidean distances from p1 points to p2 spheres
    dist = torch.norm(dist, 2, dim=3)       # B N M

    # TODO Handle NaN values in p2 radii by setting distances to infinity
    # p2_r_mask = torch.isnan(p2_r)
    # dist_clone = torch.clone(dist)
    # dist_clone[p2_r_mask] = float('inf')

    # Find the minimum distances and their corresponding indices
    min_dist, min_indice = torch.min(dist, dim=2)  # B N
    p2p_min_dist = min_dist

    # Prepare for calculating distances to sphere surfaces
    min_indice = torch.unsqueeze(min_indice, 2)    # B N 1 
    min_dist = torch.unsqueeze(min_dist, 2)        # B N 1
    # Gather the radii of the closest spheres
    p2_min_r = torch.gather(p2_r, 2, min_indice)
    # Calculate distances from p1 to the closest sphere surfaces
    min_dist = min_dist - p2_min_r
    p2s_min_dist = torch.norm(min_dist, 2, dim=2)

    return p2p_min_dist, p2s_min_dist


def sphere2point_distance_with_batch(p1, p2):
    """
    Calculate the distances from sphere p1 to the closest points in p2.

    :param p1: size[B, N, 4] - Batch of spheres represented as (x, y, z, radius)
    :param p2: size[B, M, 3] - Batch of 3D points
    :return: p2p_min_dist: size[B, N] - Minimum distances from p1 to the closest center points in p2
             p2s_min_dist: size[B, N] - Minimum distances from p1 to the closest sphere surfaces in p2
    """

    assert p1.size(0) == p2.size(0) and p1.size(2) == 4 and p2.size(2) == 3

    # Expand dimensions for broadcasting
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    # Extract radii of p1 spheres
    p1_r = p1[:, :, :, 3]                         # B 1 N
    p1_xyz = p1[:, :, :, :3]                      # B 1 N 3

    # Repeat tensors to have compatible shapes
    p1_r = p1_r.transpose(1, 2)                   # B N 1
    p1_xyz = p1_xyz.repeat(1, p2.size(2), 1, 1)   # B M N 3
    p1 = p1_xyz.transpose(1, 2)                   # B N M 3

    p2 = p2.repeat(1, p1.size(1), 1, 1)           # B N M 3 
    dist = torch.add(p1, torch.neg(p2))           # B N M 3
    dist = torch.norm(dist, 2, dim=3)             # B N M

    min_dist, min_indice = torch.min(dist, dim=2) # B N
    p2p_min_dist = min_dist

    min_dist = torch.unsqueeze(min_dist, 2)       # B N 1
    min_dist = min_dist - p1_r
    p2s_min_dist = torch.norm(min_dist, 2, dim=2)

    return p2p_min_dist, p2s_min_dist


if __name__ == '__main__':

    torch.manual_seed(100)

    a = torch.rand(8, 16, 3)
    b = torch.rand(8, 6, 4)
    b[:2, :2, :4] = float('inf')
    b[:2, :2, 3] = 0

    # print(b)
    print('===========================================')
    p2p_min_dist, p2s_min_dist = point2sphere_distance_with_batch(a, b)

    assert (~torch.isnan(p2p_min_dist)).all(), "Found NaN in Point to Sphere"    
    assert (~torch.isnan(p2s_min_dist)).all(), "Found NaN in Point to Sphere"    

    print(torch.var(p2p_min_dist))
    print(torch.mean(p2s_min_dist))

    print('===========================================')
    p2p_min_dist, p2s_min_dist = sphere2point_distance_with_batch(b, a)
    p2p_min_dist = p2p_min_dist[~torch.isinf(p2p_min_dist)]
    p2s_min_dist = p2s_min_dist[~torch.isinf(p2s_min_dist)]

    print(torch.var(p2p_min_dist))
    print(torch.mean(p2s_min_dist))