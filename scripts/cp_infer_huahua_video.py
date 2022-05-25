import os

# 用于huahua
os.chdir("/home/dell/first/AlphaPose/")

#   --posebatch 4 --qsize 128
# CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml"
CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
directory = "/home/dell/first/AlphaPose/examples/demo/cut.mp4"
output = "/home/dell/first/AlphaPose/examples/demo/output"
CKPT = "/home/dell/first/exp/54-256x192_res50_lr1e-3_1x.yaml/final_DPG.pth"
reidimg = "/home/dell/first/AlphaPose/examples/demo/output/id/"

os.system("python /home/dell/first/AlphaPose/scripts/demo_inference.py \
    --cfg " + CONFIG + \
          " --checkpoint " + CKPT + \
          " --video " + directory + \
          " --outdir " + output + \
          "--reidimg" + reidimg + \
          " --detector yolo  --save_video  --showbox --pose_track --vis_fast --posebatch 4 --sp --qsize 20")

