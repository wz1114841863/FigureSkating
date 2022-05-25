import os

# 用于huahua
os.chdir("/home/dell/first/AlphaPose/")

CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
directory = "/home/dell/first/Dataset/Huahua/test/"
output = "/home/dell/first/exp/output/exp-"

for i in os.listdir("/home/dell/first/exp/"):
    if i == "output":
        continue

    CKPT = "/home/dell/first/exp/" + i + "/final_DPG.pth"
    head = i.split('-')[0]

    # if head != "17":
    if int(head) < 50:
        continue
    os.system("python /home/dell/first/AlphaPose/scripts/demo_inference.py \
        --cfg " + CONFIG + \
              " --checkpoint " + CKPT + \
              " --indir " + directory + \
              " --outdir " + output + head + \
              " --detector yolo")
# os.system("python /home/dell/first/scores/huahua/score.py")
