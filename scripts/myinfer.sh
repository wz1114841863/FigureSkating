# CONFIG="/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
# CKPT="/home/dell/first/AlphaPose/pretrained_models/fast_res50_256x192.pth"
# img_directory="/home/dell/first/AlphaPose/examples/demo/11.mp4"
# output_directory="/home/dell/first/AlphaPose/examples/output_old"

# python scripts/demo_inference.py \
#     --cfg ${CONFIG} \
#     --checkpoint ${CKPT} \
#     --video ${img_directory} \
#     --outdir ${output_directory} \
#     --detector yolo \
#     --save_video
cd "/home/dell/first/AlphaPose/"
CONFIG="/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
OCKPT="/home/dell/first/AlphaPose/pretrained_models/fast_res50_256x192.pth"
img_directory="/home/dell/first/Dataset/Tice/test/"
output_directory="/home/dell/first/exp/output/exp-0-bright/"

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${OCKPT} \
    --indir ${img_directory} \
    --outdir ${output_directory} \
    --detector yolo
