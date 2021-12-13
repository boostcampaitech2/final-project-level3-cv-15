PARTITION=$1
CHECKPOINT_DIR=$2

echo 'configs/hrnet/fcn_hr18s_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION fcn_hr18s_512x512_160k_ade20k configs/hrnet/fcn_hr18s_512x512_160k_ade20k.py $CHECKPOINT_DIR/fcn_hr18s_512x512_160k_ade20k_20200614_214413-870f65ac.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/fcn_hr18s_512x512_160k_ade20k --options dist_params.port=28171 &
echo 'configs/hrnet/fcn_hr18s_512x1024_160k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION fcn_hr18s_512x1024_160k_cityscapes configs/hrnet/fcn_hr18s_512x1024_160k_cityscapes.py $CHECKPOINT_DIR/fcn_hr18s_512x1024_160k_cityscapes_20200602_190901-4a0797ea.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/fcn_hr18s_512x1024_160k_cityscapes --options dist_params.port=28172 &
echo 'configs/hrnet/fcn_hr48_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION fcn_hr48_512x512_160k_ade20k configs/hrnet/fcn_hr48_512x512_160k_ade20k.py $CHECKPOINT_DIR/fcn_hr48_512x512_160k_ade20k_20200614_214407-a52fc02c.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/fcn_hr48_512x512_160k_ade20k --options dist_params.port=28173 &
echo 'configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION fcn_hr48_512x1024_160k_cityscapes configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py $CHECKPOINT_DIR/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/fcn_hr48_512x1024_160k_cityscapes --options dist_params.port=28174 &
echo 'configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION pspnet_r50-d8_512x1024_80k_cityscapes configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py $CHECKPOINT_DIR/pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/pspnet_r50-d8_512x1024_80k_cityscapes --options dist_params.port=28175 &
echo 'configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION pspnet_r101-d8_512x1024_80k_cityscapes configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py $CHECKPOINT_DIR/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/pspnet_r101-d8_512x1024_80k_cityscapes --options dist_params.port=28176 &
echo 'configs/pspnet/pspnet_r101-d8_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION pspnet_r101-d8_512x512_160k_ade20k configs/pspnet/pspnet_r101-d8_512x512_160k_ade20k.py $CHECKPOINT_DIR/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/pspnet_r101-d8_512x512_160k_ade20k --options dist_params.port=28177 &
echo 'configs/pspnet/pspnet_r50-d8_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION pspnet_r50-d8_512x512_160k_ade20k configs/pspnet/pspnet_r50-d8_512x512_160k_ade20k.py $CHECKPOINT_DIR/pspnet_r50-d8_512x512_160k_ade20k_20200615_184358-1890b0bd.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/pspnet_r50-d8_512x512_160k_ade20k --options dist_params.port=28178 &
echo 'configs/resnest/pspnet_s101-d8_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION pspnet_s101-d8_512x512_160k_ade20k configs/resnest/pspnet_s101-d8_512x512_160k_ade20k.py $CHECKPOINT_DIR/pspnet_s101-d8_512x512_160k_ade20k_20200807_145416-a6daa92a.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/pspnet_s101-d8_512x512_160k_ade20k --options dist_params.port=28179 &
echo 'configs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION pspnet_s101-d8_512x1024_80k_cityscapes configs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes.py $CHECKPOINT_DIR/pspnet_s101-d8_512x1024_80k_cityscapes_20200807_140631-c75f3b99.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/pspnet_s101-d8_512x1024_80k_cityscapes --options dist_params.port=28180 &
echo 'configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION fast_scnn_lr0.12_8x4_160k_cityscapes configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py $CHECKPOINT_DIR/fast_scnn_8x4_160k_lr0.12_cityscapes-0cec9937.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/fast_scnn_lr0.12_8x4_160k_cityscapes --options dist_params.port=28181 &
echo 'configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION deeplabv3plus_r101-d8_769x769_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py $CHECKPOINT_DIR/deeplabv3plus_r101-d8_769x769_80k_cityscapes_20200607_000405-a7573d20.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/deeplabv3plus_r101-d8_769x769_80k_cityscapes --options dist_params.port=28182 &
echo 'configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION deeplabv3plus_r101-d8_512x1024_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py $CHECKPOINT_DIR/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/deeplabv3plus_r101-d8_512x1024_80k_cityscapes --options dist_params.port=28183 &
echo 'configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION deeplabv3plus_r50-d8_512x1024_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py $CHECKPOINT_DIR/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/deeplabv3plus_r50-d8_512x1024_80k_cityscapes --options dist_params.port=28184 &
echo 'configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION deeplabv3plus_r50-d8_769x769_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py $CHECKPOINT_DIR/deeplabv3plus_r50-d8_769x769_80k_cityscapes_20200606_210233-0e9dfdc4.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/deeplabv3plus_r50-d8_769x769_80k_cityscapes --options dist_params.port=28185 &
echo 'configs/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION upernet_vit-b16_ln_mln_512x512_160k_ade20k configs/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k.py $CHECKPOINT_DIR/upernet_vit-b16_ln_mln_512x512_160k_ade20k-f444c077.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/upernet_vit-b16_ln_mln_512x512_160k_ade20k --options dist_params.port=28186 &
echo 'configs/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION upernet_deit-s16_ln_mln_512x512_160k_ade20k configs/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k.py $CHECKPOINT_DIR/upernet_deit-s16_ln_mln_512x512_160k_ade20k-c0cd652f.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/upernet_deit-s16_ln_mln_512x512_160k_ade20k --options dist_params.port=28187 &
echo 'configs/deeplabv3plus/deeplabv3plus_r101-d8_fp16_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION deeplabv3plus_r101-d8_fp16_512x1024_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r101-d8_fp16_512x1024_80k_cityscapes.py $CHECKPOINT_DIR/deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes-cc58bc8d.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes --options dist_params.port=28188 &
echo 'configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 tools/slurm_test.sh $PARTITION upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py $CHECKPOINT_DIR/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth --eval mIoU --work-dir work_dirs/benchmark_evaluation/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K --options dist_params.port=28189 &
