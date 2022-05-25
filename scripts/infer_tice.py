import os

# 用于体测
os.chdir("/home/dell/first/AlphaPose/")

CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
CKPT = "/home/dell/first/exp/2-256x192_res50_lr1e-3_1x.yaml/model_299.pth"
directory = "/home/dell/first/Dataset/Tice/test/"
output = "/home/dell/first/exp/output/exp-"

for i in os.listdir("/home/dell/first/exp/"):

    if i == "output":
        continue

    if i == "0-pretrain":
        CKPT = "/home/dell/first/AlphaPose/pretrained_models/fast_res50_256x192.pth"
    else:
        CKPT = "/home/dell/first/exp/" + i + "/final_DPG.pth"

    if i.split('-')[0] != "17":
        continue
    os.system("python /home/dell/first/AlphaPose/scripts/demo_inference.py \
        --cfg " + CONFIG + \
              " --checkpoint " + CKPT + \
              " --indir " + directory + \
              " --outdir " + output + i.split('-')[0] + \
              " --detector yolo")
os.system("python /home/dell/first/scores/tice/score.py")
