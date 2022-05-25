from fastapi import FastAPI, Body
import os
import time
from celery.result import AsyncResult
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis
from redis_base.cache_redis import redis_pool, init_redis
import sys
sys.path.append('/home/dell/wz/AlphaPose/')
sys.path.append('/home/dell/zhw/MHFormer/')
sys.path.append('/home/dell/gzl/mmaction2/')
sys.path.append('/home/dell/wz/AlphaPose/server/skating2/')
sys.path.append('/home/dell/wz/AlphaPose/server/skating2/redis_base/')
sys.path.append('/home/dell/wz/AlphaPose/server')
from req import send_output, send_ready_signal
# name_json保存的add
name_json_add = '/home/dell/wz/AlphaPose/server/test.json'
# 后端api
main_ip = '10.112.6.220'
main_port = 4013
# 模型config
folder_dir = "/home/dell/wz/AlphaPose/"

CONFIG = "/home/dell/wz/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
directory = "/home/dell/wz/AlphaPose/examples/video/04.mp4"
output = "/home/dell/wz/AlphaPose/examples/demo/output"
CKPT = "/home/dell/wz/exp/54-256x192_res50_lr1e-3_1x.yaml/final_DPG.pth"
reidimg = "/home/dell/wz/AlphaPose/examples/demo/output/id/"

output_url = 'http://10.112.6.220:8888/video/pushVideoJsonInfo'

video_intput_path = "/home/dell/wz/AlphaPose/examples/demo/output/intput_tmp.mp4"

from pydantic import BaseModel

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
    os.system("nohup python /home/dell/wz/AlphaPose/server/skating2/websocket_api.py &")
    os.system("nohup python /home/dell/wz/AlphaPose/server/skating2/stream_api.py &")
    print("Receive the ready signal!")


def alphapose(mp4directory):
    # 用于huahua
    os.chdir("/home/dell/wz/AlphaPose/")
    #   --posebatch 4 --qsize 128
    # CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml"
    CONFIG = "/home/dell/wz/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
    directory = "/home/dell/skating_front/download/test.mp4"
    output = "/home/dell/wz/AlphaPose/examples/output/"

    CKPT = "/home/dell/wz/AlphaPose/pth/final_DPG.pth"
    reidimg = "/home/dell/wz/AlphaPose/examples/demo/output/reid/"

    #ffmp = "ffmpeg -i " + "/home/dell/wz/AlphaPose/examples/video/peng_jin-2022-01-17.mp4" + " -b:v 5M -vcodec mpeg4 -y -v quiet " + "/home/dell/wz/AlphaPose/examples/video/peng_jin-2022-01-17_1.mp4"
    #os.system(ffmp)

    os.system("CUDA_VISIBLE_DEVICES=3 nohup python /home/dell/wz/AlphaPose/scripts/demo_inference_cjm.py" + \
        " --cfg " + CONFIG + \
        " --checkpoint " + CKPT + \
        " --video " + directory + \
        " --outdir " + output + \
        " --reidimg " + reidimg + \
        #" --pose_track " + \
        " --detector yolo --save_video --flip --showbox --vis_fast --qsize 40 > /home/dell/wz/AlphaPose/alphapose/logs/alphapose.out 2>&1 &")
    print("alphapose start!")
    
def mhformer(mp4directory):
    # 用于huahua
    #os.chdir("/home/dell/xcy/mm_pose/mmpose/")
    #mp4directory=directory
    print('3d:',mp4directory)
    
    os.system("nohup python /home/dell/zhw/MHFormer/demo/vis_batchsize_redis.py  --video /home/dell/skating_front/download/test.mp4 --gpu 2 > /home/dell/wz/AlphaPose/alphapose/logs/mhformer.out 2>&1 &")
    print("3d start!")
    
def mmaction(mp4directory):
    # 用于huahua
    #os.chdir("/home/dell/gzl/mmaction2/")
    #mp4directory=directory
    print('mmaction:',mp4directory)
    
    os.system("CUDA_VISIBLE_DEVICES=1 nohup python /home/dell/gzl/mmaction2/project/show_into_redis.py > /home/dell/wz/AlphaPose/alphapose/logs/mmaction2.out 2>&1 &")
    print("mmaction start!")

def delete_key_redis(redis_conn, key):
    redis_conn.delete(key)

# 接收到视频地址，开始处理
@app.post("/video_upload")
def start_process(data: video_Item):
    redis_conn = redis.Redis(connection_pool= redis_pool)
    print(dict(data)['data'])
    init_redis(redis_conn)
    alphapose(dict(data)['data'])
    mhformer(dict(data)['data'])
    mmaction(dict(data)['data'])
    
    
    while redis_conn.llen('3d_video_stream') < 340:
        time.sleep(1)
    #time.sleep(30)
    send_ready_signal()
    
    os.system("nohup python /home/dell/wz/AlphaPose/server/skating2/websocket_api.py > /home/dell/wz/AlphaPose/alphapose/logs/websocket.out 2>&1 &")
    os.system("nohup python /home/dell/wz/AlphaPose/server/skating2/stream_api.py > /home/dell/wz/AlphaPose/alphapose/logs/stream.out 2>&1 &")
    print("Receive the ready signal!")
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
