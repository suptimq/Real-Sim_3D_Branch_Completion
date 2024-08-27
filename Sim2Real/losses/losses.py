import torch
import torch.nn as nn
# from pytorch3d.ops.points_normals import estimate_pointcloud_normals
# from knn_cuda import KNN

from models.Transformer_utils import *

from losses.loss_utils import point2sphere_distance_with_batch, sphere2point_distance_with_batch, closest_distance_with_batch


def compute_loss(shape_xyz, skel_xyz, skel_radius, w1, w2):
    bn = skel_xyz.size()[0]
    shape_pnum = float(shape_xyz.size()[1])
    skel_pnum = float(skel_xyz.size()[1])

    # sampling loss
    e = 0.57735027
    sample_directions = torch.tensor(
        [[e, e, e], [e, e, -e], [e, -e, e], [e, -e, -e], [-e, e, e], [-e, e, -e], [-e, -e, e], [-e, -e, -e]])
    sample_directions = torch.unsqueeze(sample_directions, 0)
    sample_directions = sample_directions.repeat(bn, int(skel_pnum), 1).cuda()
    sample_centers = torch.repeat_interleave(skel_xyz, 8, dim=1)
    sample_radius = torch.repeat_interleave(skel_radius, 8, dim=1)
    sample_xyz = sample_centers + sample_radius * sample_directions

    cd_sample1 = closest_distance_with_batch(sample_xyz, shape_xyz) / (skel_pnum * 8)
    cd_sample2 = closest_distance_with_batch(shape_xyz, sample_xyz) / (shape_pnum)
    loss_sample = cd_sample1 + cd_sample2

    # point2sphere loss
    skel_xyzr = torch.cat((skel_xyz, skel_radius), 2)
    _, tmp1 = point2sphere_distance_with_batch(shape_xyz, skel_xyzr)
    cd_point2pshere1 = torch.sum(tmp1) / shape_pnum
    _, tmp2 = sphere2point_distance_with_batch(skel_xyzr, shape_xyz)
    cd_point2sphere2 = torch.sum(tmp2) / skel_pnum
    loss_point2sphere = cd_point2pshere1 + cd_point2sphere2

    # radius loss
    loss_radius = - torch.sum(skel_radius) / skel_pnum

    # loss combination
    final_loss = loss_sample + loss_point2sphere * w1 + loss_radius * w2

    return final_loss

def compute_loss_pre(shape_xyz, skel_xyz):

    cd1 = closest_distance_with_batch(shape_xyz, skel_xyz)
    cd2 = closest_distance_with_batch(skel_xyz, shape_xyz)
    loss_cd = cd1 + cd2
    loss_cd = loss_cd * 0.0001

    return loss_cd


def compute_loss_supervision(gt_xyz, gt_radius, skel_xyz, skel_radius):

    gt_num = gt_xyz.shape[1]
    skel_pnum = skel_xyz.shape[1]
    min_dist, min_indice = closest_distance_with_batch(gt_xyz, skel_xyz, is_sum=False)
    # skeleton position loss
    cd_skel1 = torch.sum(min_dist) / gt_num
    # skeleton radius loss
    radius_diff = torch.abs(gt_radius - torch.gather(skel_radius, 1, min_indice.unsqueeze(-1)))
    cd_skelr1 = torch.sum(radius_diff) / gt_num

    min_dist2, min_indice2 = closest_distance_with_batch(skel_xyz, gt_xyz, is_sum=False)
    cd_skel2 = torch.sum(min_dist2) / skel_pnum
    radius_diff2 = torch.abs(skel_radius - torch.gather(gt_radius, 1, min_indice2.unsqueeze(-1)))
    cd_skelr2 = torch.sum(radius_diff2) / skel_pnum
    
    loss_skel = cd_skel1 + cd_skel2
    loss_skelr = cd_skelr1 + cd_skelr2

    return loss_skel, loss_skelr


class CPCLoss(nn.Module):
    """
    The Cylindrical Prior Constraint
    """
    def __init__(self, alpha=[0.1, 0.1]):
        super().__init__()
        self.alpha = alpha

    def forward(self, points, centers, center_radii):

        # Radius was filled in 0s when unavailable
        segment_points = torch.concat((centers, center_radii), dim=2)

        p2p_min_dist12, p2s_min_dist12 = point2sphere_distance_with_batch(points, segment_points)
        var12 = torch.var(p2p_min_dist12)
        mean12 = torch.mean(p2s_min_dist12)

        p2p_min_dist21, p2s_min_dist21 = sphere2point_distance_with_batch(segment_points, points)
        var21 = torch.var(p2p_min_dist21)
        mean21 = torch.mean(p2s_min_dist21)

        return self.alpha[0] * (var12 + var21), self.alpha[1] * (mean12 + mean21)


class RepulsionLoss(nn.Module):
    def __init__(self, knn=8, radius=0.07, h=0.03, alpha=0.1):
        super(RepulsionLoss, self).__init__()
        self.knn = knn  # Number of neighbors to consider
        self.radius = radius  # Repulsion radius
        self.alpha = alpha  # Loss weight
        self.h = h

        # self.KNN = KNN(k=knn, transpose_mode=True)

    def forward(self, points):

        B, N, _ = points.size()
        # _, group_idx = self.KNN(points, points)
        group_idx = knn_point(self.knn, points, points)
        knn_points = index_points(points, group_idx)   # B N K C

        knn_dist = points.unsqueeze(2).expand(-1, -1, self.knn, -1) - knn_points
        knn_dist_norm = torch.linalg.norm(knn_dist, dim=-1)

        # Calculate the repulsion loss
        weight = torch.exp(-knn_dist_norm**2 / self.h**2)
        repulsion_loss = torch.mean((self.radius - knn_dist_norm)*weight)
        
        return self.alpha * repulsion_loss


class AdversarialLoss(nn.Module):
    """
    https://github.com/soumith/ganhacks/issues/36#issuecomment-492964089
    """
    def __init__(self, alpha=[0.1, 0.1]):
        super(AdversarialLoss, self).__init__()
        self.alpha = alpha
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, inp, target):

        target = torch.zeros(inp.shape, device=inp.device) + target
        loss = self.loss_func(inp, target)

        return loss


# class ROSALoss(nn.Module):
#     """
#     The Rotational Symmetrical Axis Constraint
#     """
#     def __init__(self, neighborhood_size=32):
#         super().__init__()
#         self.neighborhood_size = neighborhood_size

#     def forward(self, xyz):
#         normals = estimate_pointcloud_normals(xyz, neighborhood_size=self.neighborhood_size)


# class ManifoldnessConstraint(nn.Module):
#     """
#     The Normal Consistency Constraint
#     """
#     def __init__(self, support=8, neighborhood_size=32, alpha=0.01):
#         super().__init__()
#         self.cos = nn.CosineSimilarity(dim=3, eps=1e-6)
#         self.support = support
#         self.neighborhood_size = neighborhood_size
#         self.alpha = alpha  # Loss weight

#     def forward(self, xyz):

#         normals = estimate_pointcloud_normals(xyz, neighborhood_size=self.neighborhood_size)

#         # idx = pointops.knn(xyz, xyz, self.support)[0]
#         # neighborhood = pointops.index_points(normals, idx)
#         idx = knn_point(self.support, xyz, xyz)
#         neighborhood = index_points(normals, idx)

#         cos_similarity = self.cos(neighborhood[:, :, 0, :].unsqueeze(2), neighborhood)
#         penalty = 1 - cos_similarity
#         penalty = penalty.std(-1)
#         penalty = penalty.mean(-1)
#         return self.alpha * penalty