import requests
import json

# 前端接受图片的url
image_url = 'http://10.112.6.220:8888/video/pushImages'
# 前端接受输出的url
output_url = 'http://10.112.6.220:8888/video/pushVideoJsonInfo'
# name_json保存的add
name_json_add = '/home/dell/wz/AlphaPose/server/test.json'
# 后端api
main_ip = 'localhost'
main_port = 4013
# 模型config
folder_dir = "/home/dell/wz/AlphaPose/"

CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
directory = "/home/dell/first/AlphaPose/examples/demo/cut.mp4"
output = "/home/dell/first/AlphaPose/examples/demo/output"
CKPT = "/home/dell/first/exp/54-256x192_res50_lr1e-3_1x.yaml/final_DPG.pth"
import os


def send_image(img0='/home/dell/first/AlphaPose/examples/demo/output/id/1.jpg',
               img1='/home/dell/first/AlphaPose/examples/demo/output/id/2.jpg'):
    try:
        # r = requests.post(image_url + "?data=" + str({0:img0,1:img1}))
        # print("post :" + image_url + "?data=" + str({0:img0,1:img1}), r)
        if 'test.json' in os.listdir(name_json_add[:-9]):
            os.system("rm " + name_json_add)
        r = requests.post(image_url, json={"0": img0, "1": img1})
        str = {"0": img0, "1": img1}
        print(f"请求内容：{str}")
        print(f"图片请求状态信号：{r}")
    except Exception as msg:
        print("上传失败:%s" % str(msg))
        return ""


def send_ready_signal():
    signal_url = 'http://10.112.6.220:8887/video/pushVideoReady'
    try:
        r = requests.get(signal_url)
        print("发送准备信号.")
    except Exception as msg:
        print("发送准备信号失败:%s" % str(msg))
        return ""

def send_output(video_add=output + '/' + 'output.mp4', json_add=output + '/' + 'output.json'):
    # r = requests.post(output_url+ "?data=" +str({"video": video_add,"json": json_add}))
    r = requests.post(output_url, json={"video": video_add, "json": json_add})
    print(f"最终结果请求的状态信息：{r}")


def return_name():
    import time
    while True:
        if 'test.json' in os.listdir(name_json_add[:-9]):
            time.sleep(0.5)
            break
    with open(name_json_add, "r+", encoding='utf-8_sig') as f:
        line = f.read()
        # print(f"替换前的line长度{len(line)}")
        # print(f"替换前的line{line} \n")
        line_re = line.replace(r"'", r'"')
        # print(f"替换后的line_re{line_re} \n")
        data = json.loads(line_re)
        print(f"data:{data}")
    return data
