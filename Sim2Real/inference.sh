export CUDA_VISIBLE_DEVICES=6

python tools/inference.py \
        cfgs/Tree_models/AdaPoinTr.yaml checkpoints/AdaPoinTr_Skeleton.pth \
        --save_vis_img  \
        --category 13124818 \
        --out_pc_root inference/ \
        --pc_root 