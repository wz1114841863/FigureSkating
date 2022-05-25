import os

# 用于huahua
os.chdir("/home/dell/wz/AlphaPose/")



#   --posebatch 4 --qsize 128
# CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml"
CONFIG = "/home/dell/wz/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
directory = "/home/dell/wz/AlphaPose/examples/video/peng_jin-2022-01-17_1.mp4"
output = "/home/dell/wz/AlphaPose/examples/output/"

CKPT = "/home/dell/wz/AlphaPose/pth/final_DPG.pth"
reidimg = "/home/dell/wz/AlphaPose/examples/demo/output/reid/"

ffmp = "ffmpeg -i " + "/home/dell/wz/AlphaPose/examples/video/peng_jin-2022-01-17.mp4" + " -b:v 5M -vcodec mpeg4 -y -v quiet " + "/home/dell/wz/AlphaPose/examples/video/peng_jin-2022-01-17_1.mp4"
os.system(ffmp)

import time
start_time = time.time()
os.system("CUDA_VISIBLE_DEVICES=2 python /home/dell/wz/AlphaPose/scripts/demo_inference.py" + \
        " --cfg " + CONFIG + \
        " --checkpoint " + CKPT + \
        " --video " + directory + \
        " --outdir " + output + \
        " --reidimg " + reidimg + \
        #" --pose_track " + \
        " --detector yolo --save_video --showbox --sp --vis_fast --qsize 20")
end_time = time.time()
print(f"总计所耗费时间{end_time - start_time}s")