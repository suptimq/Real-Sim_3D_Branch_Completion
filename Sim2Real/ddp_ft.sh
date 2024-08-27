export CUDA_VISIBLE_DEVICES=6,7

bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/Tree_models/AdaPoinTr.yaml \
    --exp_dir ./work_dirs \
    --exp_name test_poinTr_exp \
    --start_ckpts checkpoints/AdaPoinTr_Skeleton.pth \
    --train_vis_freq 20 \
    --vis_freq 100 \
    # --resume \