import requests
import json
from fastapi import FastAPI, Body
import os
from req import send_output, send_ready_signal
import time
import cv2
from celery.result import AsyncResult
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# name_json保存的add
name_json_add = '/home/dell/wz/AlphaPose/server/test.json'
# 后端api
main_ip = '10.112.6.220'
main_port = 4013
# 模型config
folder_dir = "/home/dell/wz/AlphaPose/"

CONFIG = "/home/dell/wz/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
directory = "/home/dell/wz/AlphaPose/examples/demo/input.mp4"
output = "/home/dell/wz/AlphaPose/examples/demo/output"
CKPT = "/home/dell/wz/exp/54-256x192_res50_lr1e-3_1x.yaml/final_DPG.pth"
reidimg = "/home/dell/wz/AlphaPose/examples/demo/output/id/"

output_url = 'http://10.112.6.220:8888/video/pushVideoJsonInfo'

video_intput_path = "/home/dell/wz/AlphaPose/examples/demo/output/intput_tmp.mp4"

from pydantic import BaseModel

from multiprocessing import Queue

class name_Item(BaseModel):
    name: str
    sex: str

class user_Item(BaseModel):
    user_name: str
    user_id: str


class video_Item(BaseModel):
    data: str


app = FastAPI()

templates = Jinja2Templates(directory="templates")

origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from multiprocessing import Process, Queue


# 写数据进程执行的代码
def write(q, result, index = None):
    print('write task:{}'.format(index))
    q.put(str(result[index]))
    print( 'write task:{} done'.format( index ) )
    
# 准备信号测试
@app.get("/video/test_request")
def test_req():
    send_ready_signal()
    

@app.get("/video/push_signal")
def start_process():
    os.system("nohup python websocket_api.py &")
    os.system("nohup python stream_api.py &")
    print("Receive the ready signal!")


def alphapose(mp4directory=directory):
    # 用于huahua
    os.chdir(folder_dir)
    
    #   --posebatch 4 --qsize 128
    # CONFIG = "/home/dell/wz/AlphaPose/configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml"
    # 修改输入视频格式
    ffmp = "ffmpeg -i " + "/home/dell/wz/AlphaPose/server/peng_jin-2022-01-17.mp4" + " -b:v 5M -vcodec mpeg4 -y -v quiet " + video_intput_path
    os.system(ffmp)
    #
    os.system("python /home/dell/wz/AlphaPose/scripts/demo_inference.py \
               --cfg " + CONFIG + \
              " --checkpoint " + CKPT + \
              " --video " + video_intput_path + \
              " --outdir " + output + \
              " --detector yolo  --save_video  --showbox --pose_track --vis_fast --posebatch 4 --sp --qsize 20")



# 接收到视频地址，开始处理
@app.post("/video_upload")
def start_process(data: video_Item):
    print(dict(data)['data'])

    alphapose(dict(data)['data'])
    print("complete alphapose!")
    send_output()
    # video_add = output + '/' + 'output.mp4'
    # json_add = output + '/' + 'output.json'
    # requests.post(output_url, json={"video": video_add, "json": json_add})
    # send_image("/home/dell/wz/server/9.jpg", "/home/dell/wz/server/83.jpg")
    # print(data)
    # print(return_name())

    

if __name__ == '__main__':
    import uvicorn
    import nest_asyncio

    nest_asyncio.apply()
    uvicorn.run(app, host=main_ip, port=main_port)
