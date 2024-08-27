@REM echo off

@REM python tools/inference.py ^
@REM cfgs/Skeleton_models/AdaPoinTr.yaml ^
@REM E:\Weight\Discriminator-AdaPoinTr-Skeleton-GAN_LTB81-NoTaper_CDL1_Repulsion_SkelLoss-Supervised-1\ckpt-best.pth ^
@REM --center --center_back --save_vis_img  --category 13124818 ^
@REM --pc_root D:\Data\Apple_Orchard\Lailiang_Cheng\LLC_02022022\Row13_Raw ^
@REM --out_pc_root E:\Result\LLC_02022022\Row13\Discriminator-AdaPoinTr-Skeleton-GAN_LTB81-NoTaper_CDL1_Repulsion_SkelLoss-Supervised-1\All


@echo off
setlocal EnableDelayedExpansion

rem Create a list
@REM set "items=Generator2-AdaPoinTr-Skeleton-GAN_LTB81_CDL1_Repulsion_SkelLoss-Supervised-1_Finetune Generator2-AdaPoinTr-Skeleton-GAN_FTB55_CDL1_Repulsion_SkelLoss-Supervised-1_Finetune Generator2-AdaPoinTr-Skeleton-GAN_LTB81-NoTaper_CDL1_Repulsion_SkelLoss-Supervised-1_Finetune"
set "items=Generator2-AdaPoinTr-Skeleton-GAN_FTB55-v2_CDL1_SkelLoss-Coordinate-Only-Supervised-0.01_Finetune Generator2-AdaPoinTr-Skeleton-GAN_FTB55-v2_CDL1_SkelLoss-Coordinate-Only-Supervised-0.01_Repulsion_CPC-2nd-Stage_Finetune"

rem Loop through the list
for %%i in (%items%) do (
    @REM echo Processing item: %%i
    @REM python tools/inference.py ^
    @REM cfgs/Skeleton_models/AdaPoinTr.yaml ^
    @REM E:\Weight\%%i\ckpt-best.pth ^
    @REM --center --center_back --save_vis_img  --category 13124818 ^
    @REM --pc_root D:\Data\Apple_Orchard\Lailiang_Cheng\LLC_02022022\Row13_Raw ^
    @REM --out_pc_root E:\Result\LLC_02022022\Row13\%%i\All

    python tools/inference.py ^
    cfgs/Skeleton_models/AdaPoinTr.yaml ^
    E:\Weight\%%i\ckpt-best.pth ^
    --center --center_back --save_vis_img  --category 13124818 ^
    --pc_root E:\Data\Apple\LLC_02022022 ^
    --out_pc_root E:\Result\LLC_02022022\Row13\%%i\Primary 

    @REM python tools/inference.py ^
    @REM cfgs/Skeleton_models/AdaPoinTr.yaml ^
    @REM E:\Weight\%%i\ckpt-best.pth ^
    @REM --center --center_back --save_vis_img  --category 13124818 ^
    @REM --pc_root D:\Data\Apple_Orchard\Kenong_Xu\KNX_04042023 ^
    @REM --out_pc_root E:\Result\KNX_04042023\%%i\Primary ^
    @REM --primary_branch
)

endlocal