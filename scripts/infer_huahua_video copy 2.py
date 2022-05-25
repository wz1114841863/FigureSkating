def infer(video):
    import os

    # 用于huahua
    os.chdir("/home/dell/first/AlphaPose/")

    #   --posebatch 4 --qsize 128
    # CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml"
    CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
    directory = "/home/dell/first/AlphaPose/examples/demo/cut_video/" + video
    output = "/home/dell/first/AlphaPose/examples/demo/output/output_debug/"
    ff_in = output + 'AlphaPose_' + video
    ff_out = output + video.split('.')[0] + '.mp4'
    CKPT = "/home/dell/first/exp/54-256x192_res50_lr1e-3_1x.yaml/final_DPG.pth"
    reidimg = "/home/dell/first/AlphaPose/examples/demo/output/id/"

    os.system("python /home/dell/first/AlphaPose/scripts/demo_inference_copy.py" + \
            " --cfg " + CONFIG + \
            " --checkpoint " + CKPT + \
            " --video " + directory + \
            " --outdir " + output + \
            " --reidimg " + reidimg + \
            " --pose_track " + \
            " --detector yolo --save_video --showbox --sp --vis_fast --qsize 20")

    os.system("ffmpeg -i " + ff_in + " -b:v 5M -y -v quiet " + ff_out)
    os.system("rm " + ff_in)

if __name__ == "__main__":
    # for i in range(13):
    # 2
    for i in [101]:
        infer(str(i) + '.mkv')
