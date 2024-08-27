# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-02 14:38:36
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-03 09:23:07
# @Email:  cshzxie@gmail.com

import fpsample
import transforms3d

import numpy as np
import open3d as o3d

import torch


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            if transform.__class__ in [NormalizeObjectPose]:
                data = transform(data)
            elif transform.__class__ in [SkeletonFPS]:
                data['centers'], data['center_radii'], data['center_directions'] = transform(data['centers'], data['center_radii'], data['center_directions'])
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                            RandomMirrorPoints
                        ]:
                            data[k] = transform(v, rnd_value)
                        else:
                            data[k] = transform(v)

        return data

class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class FarthestSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):

        curr = ptcloud.shape[0]
        need = self.n_points - curr
        if need < 0:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(ptcloud)
            ## TODO fps could end up providing less number of points than self.n_points
            point_cloud = point_cloud.farthest_point_down_sample(num_samples=self.n_points)

        if need < 0:
            ptcloud = np.asarray(point_cloud.points)
            curr = ptcloud.shape[0]
            need = self.n_points - curr
            if need == 0:
                return ptcloud
        
        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))
        return ptcloud
    

class SkeletonFPS(object):
    """
    Class for performing farthest point sampling (FPS) on input point clouds.

    This class implements FPS to downsample point clouds to a fixed number of points.
    It takes input points, radii, and directions, and returns a subset of points, radii,
    and directions selected by the FPS algorithm.

    Args:
        parameters (dict): Dictionary containing parameters for FPS.
            - 'n_points': Number of points to select after FPS.

    Methods:
        __init__(self, parameters): Initialize the SkeletonFPS instance.
        __call__(self, xyz, rd, dirs): Perform farthest point sampling on input data.

    Example usage:
    ```
    parameters = {'n_points': 512}
    fps = SkeletonFPS(parameters)
    sampled_xyz, sampled_rd, sampled_dirs = fps(xyz, rd, dirs)
    ```
    """

    def __init__(self, parameters):
        """
        Initialize the SkeletonFPS instance.

        Args:
            parameters (dict): Dictionary containing parameters for FPS.
                - 'n_points': Number of points to select after FPS.
        """
        self.n_points = parameters['n_points']

    def __call__(self, xyz, rd, dirs):
        """
        Perform farthest point sampling (FPS) on input data.

        Args:
            xyz (torch.Tensor): Input point cloud coordinates. Shape: N, 3.
            rd (torch.Tensor): Input point cloud radii. Shape: N.
            dirs (torch.Tensor): Input point cloud directions. Shape: N, 3.

        Returns:
            torch.Tensor: Downsampled point coordinates after FPS. Shape: n_points, 3.
            torch.Tensor: Downsampled point radii after FPS. Shape: n_points.
            torch.Tensor: Downsampled point directions after FPS. Shape: n_points, 3.
        """

        assert xyz.shape[-1] == 3, f"Incorrect shape {xyz.shape}. Last dimension should be 3!"

        fps_samples_idx = fpsample.fps_sampling(xyz, self.n_points)

        return xyz[fps_samples_idx, :], rd[fps_samples_idx], dirs[fps_samples_idx, :]


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud

class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud

class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data
