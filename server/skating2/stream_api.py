from typing import List
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import time
import requests
import json
from fastapi import FastAPI, Body
import os
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
import redis
from redis_base.cache_redis import redis_pool, wirte_into_queue, get_from_queue, list_is_empty


app = FastAPI()
redis_conn = redis.Redis(connection_pool= redis_pool)

ensumble = False

def init_into_redis(redis_conn, key):
    cap1 = cv2.VideoCapture(r"/home/dell/wz/AlphaPose/server/peng_jin-2022-01-17.mp4")
    while True:
        res,frame = cap1.read()
        if not res:
            break
        jpeg = cv2.imencode('.jpg', frame)
        wirte_into_queue(redis_conn, key, jpeg.tobytes())
        
def delete_key_redis(redis_conn, key):
    redis_conn.delete(key)
    
def test_redis(key):
    delete_key_redis(redis_conn, key)
    #init_into_redis(redis_conn, key)

def Get_frame():
    #test_redis('main_video_stream')
    fps = 25
    interval = 0
    while True:
        #import pdb;pdb.set_trace()
        time.sleep(interval)
        t0 = time.time()
        
        if not list_is_empty(redis_conn, 'main_video_stream'):
            jpeg = get_from_queue(redis_conn, 'main_video_stream')
            #print(type(jpeg))
            jpeg = eval(jpeg)
            interval = (1/fps)-(time.time()-t0)
            interval = interval if interval>0 else 0
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            break
        
def Get_frame_heatmap():
    #test_redis('heatmap_video_stream')
    fps = 25
    interval = 0
    while True:
        #import pdb;pdb.set_trace()
        time.sleep(interval)
        t0 = time.time()
        
        if not list_is_empty(redis_conn, 'heatmap_video_stream'):
            jpeg = get_from_queue(redis_conn, 'heatmap_video_stream')
            #print(type(jpeg))
            jpeg = eval(jpeg)
            interval = (1/fps)-(time.time()-t0)
            interval = interval if interval>0 else 0
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            break
        
def Get_frame_3d():
    #test_redis('3d_video_stream')
    fps = 25
    interval = 0
    while True:
        #import pdb;pdb.set_trace()
        time.sleep(interval)
        t0 = time.time()
        if not list_is_empty(redis_conn, '3d_video_stream'):
            jpeg = get_from_queue(redis_conn, '3d_video_stream')
            #print(type(jpeg))
            jpeg = eval(jpeg)
            interval = (1/fps)-(time.time()-t0)
            interval = interval if interval>0 else 0
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            break
        
#--------------------------------------------------------------------------------------------
# 从视频源文件中读取视频流
def get_frame():
    cap1 = cv2.VideoCapture(r"/home/dell/wz/AlphaPose/examples/demo/kps_video.mp4")
    fps, size, fNUMS = get_capture_attr(cap1)
    delete_key_redis(redis_conn, 'main_video_stream')
    print(fps)
    cnt = 0
    interval = 0
    while True:
        #import pdb;pdb.set_trace()
        time.sleep(interval)
        t0 = time.time()
        res,frame = cap1.read()
        
        cnt = cnt + 1
        if cnt == 2:
            cnt = 0
            if not res:
                break
            frame = cv2.resize(frame,(640,480))
            ret, jpeg = cv2.imencode('.jpg', frame)
            wirte_into_queue(redis_conn, 'main_video_stream', jpeg.tobytes())
            jpeg = eval(get_from_queue(redis_conn, 'main_video_stream'))
            #jpeg = jpeg.tobytes()
            interval = (1/fps)-(time.time()-t0)
            interval = interval if interval>0 else 0
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            interval = (1/fps)-(time.time()-t0)
            interval = interval if interval>0 else 0
            continue
        
# 发送分析结果
def get_frame_heatmap():
    cap1 = cv2.VideoCapture(r"/home/dell/wz/AlphaPose/examples/demo/pure_heatmap.mp4")
    fps, size, fNUMS = get_capture_attr(cap1)
    delete_key_redis(redis_conn, 'heatmap_video_stream')
    print(fps)
    interval = 0
    cnt = 0
    while True:
        #import pdb;pdb.set_trace()
        time.sleep(interval)
        t0 = time.time()
        res,frame = cap1.read()
        
        cnt = cnt + 1
        if cnt == 1:
            cnt = 0
            if not res:
                break
            frame = cv2.resize(frame,(340,250))
            ret, jpeg = cv2.imencode('.jpg', frame)
            wirte_into_queue(redis_conn, 'heatmap_video_stream', jpeg.tobytes())
            jpeg = eval(get_from_queue(redis_conn, 'heatmap_video_stream'))
            #jpeg = jpeg.tobytes()
            interval = (1/fps)-(time.time()-t0)
            interval = interval if interval>0 else 0
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            interval = (1/fps)-(time.time()-t0)
            interval = interval if interval>0 else 0
            continue
        
# 发送分析结果
def get_frame_3d():
    cap1 = cv2.VideoCapture(r"/home/dell/wz/AlphaPose/examples/demo/vis_16.mp4")
    fps, size, fNUMS = get_capture_attr(cap1)
    delete_key_redis(redis_conn, '3d_video_stream')
    print(fps)
    interval = 0
    cnt = 0
    while True:
        #import pdb;pdb.set_trace()
        time.sleep(interval)
        t0 = time.time()
        res,frame = cap1.read()
        
        cnt = cnt + 1
        if cnt == 1:
            cnt = 0
            if not res:
                break
            frame = cv2.resize(frame,(340,250))
            ret, jpeg = cv2.imencode('.jpg', frame)
            wirte_into_queue(redis_conn, '3d_video_stream', jpeg.tobytes())
            jpeg = eval(get_from_queue(redis_conn, '3d_video_stream'))
            #jpeg = jpeg.tobytes()
            interval = (1/fps)-(time.time()-t0)
            interval = interval if interval>0 else 0
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            interval = (1/fps)-(time.time()-t0)
            interval = interval if interval>0 else 0
            continue
#--------------------------------------------------------------------------------------------


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

# 1.主结果视频
@app.get('/video_feed_main_result', response_class=HTMLResponse)
async def video_feed():
    return  StreamingResponse(Get_frame(),
                    media_type='multipart/x-mixed-replace; boundary=frame')

# 2.热力图视频
@app.get('/video_feed_heatmap', response_class=HTMLResponse)
async def video_feed():
    return  StreamingResponse(Get_frame_heatmap(),
                    media_type='multipart/x-mixed-replace; boundary=frame')

# 3.3d模型视频
@app.get('/video_feed_3d', response_class=HTMLResponse)
async def video_feed():
    return  StreamingResponse(Get_frame_3d(),
                    media_type='multipart/x-mixed-replace; boundary=frame')

        
def start_stream():
    import uvicorn
    # 官方推荐是用命令后启动 uvicorn main:app --host=127.0.0.1 --port=8010 --reload
    print('start stream!')
    uvicorn.run(app='stream_api:app', host="10.112.6.220", port=4012, reload=True, debug=True)
        


if __name__ == "__main__":
    import uvicorn
    # 官方推荐是用命令后启动 uvicorn main:app --host=127.0.0.1 --port=8010 --reload
    uvicorn.run(app='stream_api:app', host="10.112.6.220", port=4012, reload=True, debug=True)
