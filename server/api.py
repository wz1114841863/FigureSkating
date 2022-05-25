import requests
import json
from fastapi import FastAPI, Body
import os
from req import send_output, send_ready_signal
import time
import cv2
from celery.result import AsyncResult
from fastapi import Body, FastAPI, Form, Request,Header,Response
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# name_json保存的add
name_json_add = '/home/dell/first/AlphaPose/server/test.json'
# 后端api
main_ip = '10.112.6.220'
main_port = 4010
# 模型config
folder_dir = "/home/dell/first/AlphaPose/"

CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
directory = "/home/dell/first/AlphaPose/examples/demo/input.mp4"
output = "/home/dell/first/AlphaPose/examples/demo/output"
CKPT = "/home/dell/first/exp/54-256x192_res50_lr1e-3_1x.yaml/final_DPG.pth"
reidimg = "/home/dell/first/AlphaPose/examples/demo/output/id/"

output_url = 'http://10.112.204.127:8888/video/pushVideoJsonInfo'

video_intput_path = "/home/dell/first/AlphaPose/examples/demo/output/intput_tmp.mp4"

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

class ConnectionManager:
    def __init__(self):
        # 存放激活的ws连接对象
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        # 等待连接
        await ws.accept()
        # 存储ws连接对象
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        # 关闭时 移除ws对象
        self.active_connections.remove(ws)

    @staticmethod
    async def send_personal_message(message: str, ws: WebSocket):
        # 发送个人消息
        await ws.send_text(message)

    async def broadcast(self, message: str):
        # 广播消息
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

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

action_queue = Queue(15)
confidence_queue = Queue(15)
pose_queue = Queue(15)


send_action = True
send_confidence = True
send_pose = True

action_result = {0: '接续步',
                 1: '接续步',
                 2: '接续步',
                 3: '接续步',
                 4: '接续步',
                 5: '接续步',
                 6: '接续步',
                 7: '联合跳',
                 8: '联合跳',
                 9: '联合跳',
                10: '联合跳',
                11: '联合跳',
                12: '接续步',
                13: '接续步'
                 }
confidence_result = {0: {1,0,0,0,0,0,0,0,0,0},
                     1: {1,0,0,0,0,0,0,0,0,0},
                     2: {1,0,0,0,0,0,0,0,0,0},
                     3: {1,0,0,0,0,0,0,0,0,0},
                     4: {1,0,0,0,0,0,0,0,0,0},
                     5: {1,0,0,0,0,0,0,0,0,0},
                     6: {1,0,0,0,0,0,0,0,0,0},
                     7: {1,0,0,0,0,0,0,0,0,0},
                     8: {0,0,0,1,0,0,0,0,0,0},
                     9: {0,0,0,1,0,0,0,0,0,0},
                     10: {0,0,0,1,0,0,0,0,0,0},
                     11: {0,0,0,1,0,0,0,0,0,0},
                     12: {1,0,0,0,0,0,0,0,0,0},
                     13: {1,0,0,0,0,0,0,0,0,0}}
pose_result = {"left_shoulder": [i for i in range(0,14)],
               "right_shoulder": [i for i in range(10,24)],
               "left_hip": [i for i in range(20,34)],
               "right_hip": [i for i in range(30,44)],
               "left_knee": [i for i in range(40,54)],
               "right_knee": [i for i in range(50,64)],
               "left_ankle": [i for i in range(60,74)],
               "right_ankle": [i for i in range(70,84)]
               }

# 写数据进程执行的代码
def write(q, result, index = None):
    print('write task:{}'.format(index))
    q.put(str(result[index]))
    print( 'write task:{} done'.format( index ) )
    
for i in range(len(action_result)):
    write(action_queue, action_result, list(action_result.keys())[i])

for i in range(len(confidence_result)):
    write(confidence_queue, confidence_result, list(confidence_result.keys())[i])

for i in range(len(pose_result['left_shoulder'])):
    temp = []
    temp.append({"left_shoulder": pose_result['left_shoulder'][i],
               "right_shoulder": pose_result['left_shoulder'][i],
               "left_hip": pose_result['left_shoulder'][i],
               "right_hip": pose_result['left_shoulder'][i],
               "left_knee": pose_result['left_shoulder'][i],
               "right_knee": pose_result['left_shoulder'][i],
               "left_ankle": pose_result['left_shoulder'][i],
               "right_ankle": pose_result['left_shoulder'][i]
               })
    write(pose_queue, temp, 0)




# 准备信号测试
@app.get("/video/test_request")
def test_req():
    send_ready_signal()
    

@app.get("/video/push_signal")
def start_process():
    print("Receive the ready signal!")


def alphapose(mp4directory=directory):
    # 用于huahua
    os.chdir(folder_dir)
    
    #   --posebatch 4 --qsize 128
    # CONFIG = "/home/dell/first/AlphaPose/configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml"
    # 修改输入视频格式
    ffmp = "ffmpeg -i " + mp4directory + " -b:v 5M -vcodec mpeg4 -y -v quiet " + video_intput_path
    os.system(ffmp)
    #
    os.system("python /home/dell/first/AlphaPose/scripts/demo_inference.py \
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
    # send_image("/home/dell/first/server/9.jpg", "/home/dell/first/server/83.jpg")
    # print(data)
    # print(return_name())

# 发送分析结果
def get_frame():
    cap1 = cv2.VideoCapture(r"/home/dell/wz/AlphaPose/server/peng_jin-2022-01-17.mp4")
    fps, size, fNUMS = get_capture_attr(cap1)
    print(fps)
    interval = 0
    global send_action
    global send_confidence
    global send_pose
    cnt = 0
    while True:
        #import pdb;pdb.set_trace()
        if cnt == fps:
            send_action = True
            send_confidence = True
            send_pose = True
            cnt = 0
        time.sleep(interval)
        cnt = cnt + 1
        t0 = time.time()
        res,frame = cap1.read()
        if not res:
            break
        ret, jpeg = cv2.imencode('.jpg', frame)
        interval = (1/fps)-(time.time()-t0)
        interval = interval if interval>0 else 0
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
# 发送分析结果
def get_frame_heatmap():
    cap1 = cv2.VideoCapture(r"/home/dell/wz/AlphaPose/server/peng_jin-2022-01-17.mp4")
    fps, size, fNUMS = get_capture_attr(cap1)
    print(fps)
    interval = 0
    while True:
        #import pdb;pdb.set_trace()
        print(interval)
        time.sleep(interval)
        t0 = time.time()
        res,frame = cap1.read()
        if not res:
            break
        ret, jpeg = cv2.imencode('.jpg', frame)
        interval = (1/fps)-(time.time()-t0)
        interval = interval if interval>0 else 0
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
# 发送分析结果
def get_frame_3d():
    cap1 = cv2.VideoCapture(r"/home/dell/wz/AlphaPose/server/peng_jin-2022-01-17.mp4")
    fps, size, fNUMS = get_capture_attr(cap1)
    print(fps)
    interval = 0
    while True:
        #import pdb;pdb.set_trace()
        time.sleep(interval)
        t0 = time.time()
        res,frame = cap1.read()
        if not res:
            break
        ret, jpeg = cv2.imencode('.jpg', frame)
        interval = (1/fps)-(time.time()-t0)
        interval = interval if interval>0 else 0
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# 1.主结果视频
@app.get('/video_feed_main_result', response_class=HTMLResponse)
async def video_feed():
    return  StreamingResponse(get_frame(),
                    media_type='multipart/x-mixed-replace; boundary=frame')

# 2.热力图视频
@app.get('/video_feed_heatmap', response_class=HTMLResponse)
async def video_feed():
    return  StreamingResponse(get_frame_heatmap(),
                    media_type='multipart/x-mixed-replace; boundary=frame')

# 3.3d模型视频
@app.get('/video_feed_3d', response_class=HTMLResponse)
async def video_feed():
    return  StreamingResponse(get_frame_3d(),
                    media_type='multipart/x-mixed-replace; boundary=frame')

# 动作识别结果接口
@app.websocket("/ws/action_classification/result_stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.send_personal_message(f"action result 后端已连接.")
    user = 'action'
    global send_action
    try:
        while True:
            time.sleep(0.9)
            data = action_queue.get()
            print(data)
            await manager.send_personal_message(data, websocket)
            send_action = False
            #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端-{user}-断开")

# 动作置信度视图，前端用于接收后端发送的动作识别数据，每接收一次，需要更新一次视图中的动作分类结果。
@app.websocket("/ws/action_confidence/result_stream")
async def websocket_endpoint2(websocket: WebSocket, confidence_queue):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    user = 'confidence'
    try:
        while not confidence_queue.empty():
            print(confidence_queue.get())
            await asyncio.sleep(1)
            #data = await websocket.receive_text()
            data = confidence_queue.get()
            await manager.send_personal_message(data, websocket)
            #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端confidence断开")

# 动作角度，前端用于接收后端发送的动作识别数据，每接收一次，需要更新一次视图中的动作分类结果。
@app.websocket("/ws/pose/result_stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    user = 'pose'
    global send_pose
    try:
        while True:
            time.sleep(1)
            #data = await websocket.receive_text()
            data = pose_queue.get()
            await manager.send_personal_message(data, websocket)
            #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端-{user}-断开")

def get_frame_from_quene(frame,fps=25):
    interval = 0
    for i in range(len(frame)):

        time.sleep(interval)
        t0 = time.time()
        ret, jpeg = cv2.imencode('.jpg', frame)
        interval = (1/fps)-(time.time()-t0)
        interval = interval if interval>0 else 0
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

def get_capture_attr(capture):
    """
    get the attribute of video capture.
    return Fps, (w,h), len(frame)
    """
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    return fps, size, fNUMS




@app.websocket("/ws/action_label_stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await manager.broadcast(f"websocket 后端已连接.")
    user = 'test'
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"你说了: {data}", websocket)
            await manager.broadcast("返回置信度数据:0,1,2,3,4,5")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"用户-{user}-离开")

    

if __name__ == '__main__':
    import uvicorn
    import nest_asyncio

    nest_asyncio.apply()
    uvicorn.run(app, host=main_ip, port=main_port)
