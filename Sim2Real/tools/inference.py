##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
import argparse
import os
import numpy as np
import cv2
import sys
import open3d as o3d
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from utils.visualization import plot_pcd_one_view
from datasets.io import IO
from datasets.data_transforms import Compose

import time
import json
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config', 
        help = 'yaml config file')
    parser.add_argument(
        'model_checkpoint', 
        help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')   
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud') 
    parser.add_argument(
        '--out_pc_root',
        type=str,
        default='',
        help='root of the output pc file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--category', type=str, help='object category code - trunk (13186713) and branch (13124818)')
    parser.add_argument(
        '--center', action='store_true', help='center obj')
    parser.add_argument(
        '--center_back', action='store_true', help='transform centered obj back')    
    parser.add_argument(
        '--normalize', action='store_true', help='normalize obj')
    parser.add_argument(
        '--primary_branch', action='store_true', help='use primary branch')
    args = parser.parse_args()

    assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args


def export_file(filename, obj, ext):
    """
        Export point cloud or mesh objects
    """
    if ext in ['.ply', '.pcd']:
        if isinstance(obj, np.ndarray):
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(obj)
            obj = point_cloud
        o3d.io.write_point_cloud(filename, obj)
    elif ext in ['.obj', '.stl']:
        o3d.io.write_triangle_mesh(filename, obj)
    elif ext == '.npy':
        assert isinstance(obj, np.ndarray)
        np.save(filename, obj)


def file_io_json(json_path, content=None, mode='r'):
    if mode == 'r':
        with open(json_path, 'r') as json_file:
            meta_dict = json.load(json_file)
        return meta_dict
    elif mode == 'w':
        assert content is not None, 'Content is None'
        with open(json_path, 'w') as meta_file:
            json.dump(content, meta_file)
        print(f'dump to {json_path}')
        return None
    else:
        raise NotImplementedError


def load_off(path):
    fopen = open(path, 'r', encoding='utf-8')
    lines = fopen.readlines()
    linecount = 0
    pts = np.zeros((1, 3), np.float64)
    faces = np.zeros((1, 3), int)
    p_num = 0
    f_num = 0

    for line in lines:
        linecount = linecount + 1
        word = line.split()

        if linecount == 1:
            continue
        if linecount == 2:
            p_num = int(word[0])
            f_num = int(word[1])
            pts = np.zeros((p_num, 3), np.float32)
            faces = np.zeros((f_num, 3), int)
        if linecount >= 3 and linecount < 3 + p_num:
            pts[linecount - 3, :] = np.float64(word[0:3])
        if linecount >= 3 + p_num:
            faces[linecount - 3 - p_num] = np.int32(word[1:4])

    fopen.close()
    return pts, faces


def save_spheres(center, radius, scale_factor, path):
    # load the vertices (sp_v) and faces (sp_f) of a sphere from an OFF file
    sp_v, sp_f = load_off('data/sphere16.off')

    with open(path, "w") as file:
        # iterate over each sphere
        for i in range(center.shape[0]):
            v, r = center[i], radius[i]  # get the center and radius of the current sphere

            # scale the sphere vertices by the radius and scaling factor, then translate them by the center
            scaled_v = sp_v * r * scale_factor
            translated_v = scaled_v + v

            # write the scaled and translated vertices to the file
            for m in range(translated_v.shape[0]):
                file.write('v ' + str(translated_v[m][0]) + ' ' + str(translated_v[m][1]) + ' ' + str(translated_v[m][2]) + '\n')

        # iterate over each sphere again to define the faces
        for m in range(center.shape[0]):
            base = m * sp_v.shape[0] + 1  # calculate the base index for the current sphere
            # write the faces using the indices of the scaled and translated vertices
            for j in range(sp_f.shape[0]):
                file.write(
                    'f ' + str(sp_f[j][0] + base) + ' ' + str(sp_f[j][1] + base) + ' ' + str(sp_f[j][2] + base) + '\n')


def load_gt_excel(filepath, sheetname='Tree1'):
    df = pd.read_excel(filepath, sheet_name=sheetname)

    return df


def inference_single(model, discriminator, pc_file_list, args, config, root=None, out_root=None, gt_excel=None):
    
    confidence_dict = {}
    # add a estimated diameter column
    if gt_excel is not None:
        gt_excel = gt_excel.assign(Estimated_Diameter=0.0)
        gt_excel['Section Index'] = gt_excel['Section Index'].astype(str)
        gt_excel['Color'] = gt_excel['Color'].astype(str)

    for i, pc_path in enumerate(pc_file_list, 1):
        
        if gt_excel is not None:
            ### ========== ###
            ### LLC format ###
            ### ========== ###
            section_idx, tag_color = pc_path.split('_')
            section_idx = str(section_idx[7:])
            tag_color = str(tag_color[:-4])
            condition = (gt_excel['Section Index'] == section_idx) & (gt_excel['Color'] == tag_color)
            
            ### ========== ###
            ### KNX format ###
            ### ========== ###
            # section_idx = pc_path[6:-4]
            # condition = gt_excel['Section Index'] == section_idx

        if root is not None:
            pc_file = os.path.join(root, pc_path)
            gt_folder = Path(root).parent / 'test' / 'complete' / args.category
            gt_filename = pc_path.split('_')[0]
            gt_filepath = gt_folder / f'{gt_filename}.pcd'
        else:
            pc_file = pc_path

        # find branch root file
        pc_root_file = pc_file[:-4] + '_Root' + pc_file[-4:]
        if args.primary_branch:
            pc_file = pc_file[:-4] + '_Primary' + pc_file[-4:]
        pc_filename = Path(pc_file).stem
        print(f'==========================================')
        print(f'Processing file-{i}: {pc_file}')
        print(f'==========================================')
        # record the start time for loading data
        load_start_time = time.time()

        # read single point cloud
        pc_ndarray = IO.get(pc_file).astype(np.float32)
        if os.path.exists(gt_filepath):
            gt_ndarray = IO.get(str(gt_filepath)).astype(np.float32)
        else:
            gt_ndarray = None

        # calculate the elapsed time for loading data
        load_elapsed_time = time.time() - load_start_time

        # transform it according to the model 
        if config.dataset.train._base_['NAME'] == 'ShapeNet' or args.normalize:
            # normalize it to fit the model on ShapeNet-55/34
            centroid = np.mean(pc_ndarray, axis=0)
            pc_ndarray = pc_ndarray - centroid
            m = np.max(np.sqrt(np.sum(pc_ndarray**2, axis=1)))
            pc_ndarray = pc_ndarray / m
        elif args.center:
            centroid = np.mean(pc_ndarray, axis=0)
            pc_ndarray = pc_ndarray - centroid

        transform = Compose([{
            'callback': 'FarthestSamplePoints',
            'parameters': {
                'n_points': 2048
            },
            'objects': ['input']
        }, {
            'callback': 'ToTensor',
            'objects': ['input']
        }])

        # record the start time for transforming data
        transform_start_time = time.time()

        pc_ndarray_normalized = transform({'input': pc_ndarray})
        pc_tensor = pc_ndarray_normalized['input']
        assert pc_tensor.shape[0] == 2048, f'#Point {pc_tensor.shape[0]} Not Match'
        # calculate the elapsed time for transforming data
        transform_elapsed_time = time.time() - transform_start_time

        # record the start time for inferencing data
        inference_start_time = time.time()

        # inference
        partial = pc_tensor.unsqueeze(0).to(args.device.lower())
        ret = model(partial)
        dense_points = ret[1].detach()
        skel_xyz = ret[2].detach() if len(ret) > 2 else None
        skel_r = ret[3].detach() if len(ret) > 2 else None

        confidence = discriminator(dense_points).detach().squeeze(0).cpu().item()
        confidence_dict[pc_filename] = confidence

        # ralculate the elapsed time for inferencing data
        inference_elapsed_time = time.time() - inference_start_time

        dense_points = dense_points.squeeze(0).cpu().numpy()
        skel_xyz = skel_xyz.squeeze(0).cpu().numpy()
        skel_r = skel_r.squeeze(0).squeeze(-1).cpu().numpy()

        # transform2 = Compose([{
        #     'callback': 'FarthestSamplePoints',
        #     'parameters': {
        #         'n_points': 1024 if args.category == '13124818' else 2048
        #     },
        #     'objects': ['input']
        # }])
        # dense_points = transform2({'input': dense_points})['input']

        if config.dataset.train._base_['NAME'] == 'ShapeNet':
            # denormalize it to adapt for the original input
            dense_points = dense_points * m
            dense_points = dense_points + centroid

            skel_xyz = skel_xyz * m
            skel_xyz = skel_xyz + centroid

        if args.center_back:
            pc_partial = partial.squeeze(0).detach().cpu().numpy()
            pc_ndarray = pc_partial + centroid
            dense_points = dense_points + centroid
            skel_xyz = skel_xyz + centroid

        if os.path.exists(pc_root_file):
            pc_root_ndarray = IO.get(pc_root_file).astype(np.float32)
            mean_root_ndarray = np.mean(pc_root_ndarray, axis=0, keepdims=True)
            distances = np.linalg.norm(skel_xyz - mean_root_ndarray, axis=1)
            sorted_ind = np.argsort(distances)
            skel_xyz = skel_xyz[sorted_ind]
            skel_r = skel_r[sorted_ind]
            if gt_excel is not None:
                gt_excel.loc[condition, 'Estimated_Diameter'] = np.mean(skel_r[:3]) * 1000 * 2

        # Calculate the total elapsed time
        total_elapsed_time = time.time() - load_start_time

        print(f"Loading Time: {load_elapsed_time:.2f} seconds")
        print(f"Transforming Time: {transform_elapsed_time:.2f} seconds")
        print(f"Inferencing Time: {inference_elapsed_time:.2f} seconds")
        print(f"Total Time: {total_elapsed_time:.2f} seconds")

        if out_root != '':
            filename = os.path.splitext(pc_path)[0]
            target_path = os.path.join(out_root, filename)
            os.makedirs(target_path, exist_ok=True)

            if gt_ndarray is not None:
                export_file(os.path.join(target_path, 'gt.pcd'), gt_ndarray, '.pcd')

            export_file(f'{target_path}_fine.pcd', dense_points, '.pcd')
            # np.save(f'{target_path}_fine_confidence.npy', confidence)
            export_file(os.path.join(target_path, 'input.pcd'), pc_ndarray, '.pcd')
            export_file(os.path.join(target_path, 'fine.pcd'), dense_points, '.pcd')
            export_file(os.path.join(target_path, 'skel.pcd'), skel_xyz, '.pcd')
            save_spheres(skel_xyz, skel_r, 0.5, os.path.join(target_path, 'skel_sphere.obj'))
            file_io_json(os.path.join(target_path, 'skel_r.json'), skel_r.tolist(), mode='w')

            if args.save_vis_img:
                input_pc = pc_ndarray_normalized['input'].numpy()
                output_pc = dense_points
                gt_pc = gt_ndarray if gt_ndarray is not None else np.zeros_like(dense_points)
                plot_pcd_one_view(os.path.join(target_path, 'vis.png'), [input_pc, output_pc, gt_pc], ['Input', 'Output', 'GT'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))

                plt.hist(skel_r)
                plt.savefig(os.path.join(target_path, 'skel_r.png'))
                plt.close()

    # save excel
    if gt_excel is not None:
        gt_excel.to_csv(os.path.join(out_root, 'measurements.csv'), index=False)

    # save confidence distribution
    with open(os.path.join(out_root, 'Confidence.json'), 'w') as fp:
        json.dump(confidence_dict, fp)

    confidence_values = np.array(list(confidence_dict.values()))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
    axes[0].hist(confidence_values, color='blue', alpha=0.7)
    axes[0].set_xlabel('Confidence Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Confidence')

    confidence_norm = (confidence_values - confidence_values.min()) / (confidence_values.max() - confidence_values.min())
    axes[1].hist(confidence_norm, color='blue', alpha=0.7)
    axes[1].set_xlabel('Confidence Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Normalized Confidence')

    confidence_sigmoid = 1 / (1 + np.exp(-confidence_values)) 
    axes[2].hist(confidence_sigmoid, color='blue', alpha=0.7)
    axes[2].set_xlabel('Confidence Value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Sigmoid Confidence')

    plt.savefig(os.path.join(out_root, 'Confidence.png'))
    plt.close()

    return gt_excel

def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config)
    discriminator = builder.model_builder(config.model.discriminator)

    builder.load_model(base_model, discriminator, args.model_checkpoint)
    base_model.to(args.device.lower())
    discriminator.to(args.device.lower())
    base_model.eval()
    discriminator.eval()

    ### ========== ###
    ### LLC format ###
    ### ========== ###
    excel_filepath = r'D:\Data\Apple_Orchard\Lailiang_Cheng\Field_Measurements.xlsx'
    assert os.path.exists(excel_filepath), f'{excel_filepath} Not Found'
    
    branch_folders = [x for x in os.listdir(args.pc_root) if os.path.isdir(os.path.join(args.pc_root, x)) and x.endswith('segmented')]
    branch_excel = None

    for branch_folder in branch_folders[:]:
        out_root = os.path.join(args.out_pc_root, branch_folder)
        branch_folder = os.path.join(args.pc_root, branch_folder)
        print(branch_folder)
        tree_id = os.path.basename(branch_folder).split('_')[0]
        tree_gt_excel = load_gt_excel(excel_filepath, sheetname=tree_id.capitalize())[:-1]

        ### ========== ###
        ### LLC format ###
        ### ========== ###
        pc_file_list = [x for x in os.listdir(branch_folder) if (x.startswith('Section') or x.startswith('branch')) and x.endswith('.pcd')]
        # pc_file_list = [x for x in os.listdir(branch_folder) if x.startswith('Branch') and x.endswith('.ply') and 'Root' not in x and 'Primary' not in x]
        pc_file_list = natsorted(pc_file_list)
        tmp_excel = inference_single(base_model, discriminator, pc_file_list[:], args, config, root=branch_folder, out_root=out_root, gt_excel=None)
        branch_excel = tmp_excel if branch_excel is None else pd.concat([branch_excel, tmp_excel], ignore_index=True)
    
    if branch_excel is not None:
        branch_excel.to_csv(os.path.join(args.out_pc_root, 'measurements.csv'), index=False)

if __name__ == '__main__':
    main()