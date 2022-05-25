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
import api
from server.skating2.websocket_api import start_websocket
from server.skating2.stream_api import start_stream

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
confidence_result = {0: [1,0,0,0,0,0,0,0,0,0],
                     1: [1,0,0,0,0,0,0,0,0,0],
                     2: [1,0,0,0,0,0,0,0,0,0],
                     3: [1,0,0,0,0,0,0,0,0,0],
                     4: [1,0,0,0,0,0,0,0,0,0],
                     5: [1,0,0,0,0,0,0,0,0,0],
                     6: [1,0,0,0,0,0,0,0,0,0],
                     7: [1,0,0,0,0,0,0,0,0,0],
                     8: [0,0,0,1,0,0,0,0,0,0],
                     9: [0,0,0,1,0,0,0,0,0,0],
                     10: [0,0,0,1,0,0,0,0,0,0],
                     11: [0,0,0,1,0,0,0,0,0,0],
                     12: [0,0,0,1,0,0,0,0,0,0],
                     13: [1,0,0,0,0,0,0,0,0,0]}
pose_result = {"left_shoulder": [i for i in range(0,14)],
               "right_shoulder": [i for i in range(10,24)],
               "left_hip": [i for i in range(20,34)],
               "right_hip": [i for i in range(30,44)],
               "left_knee": [i for i in range(40,54)],
               "right_knee": [i for i in range(50,64)],
               "left_ankle": [i for i in range(60,74)],
               "right_ankle": [i for i in range(70,84)]
               }
    

if __name__=='__main__':

    p_websocket = Process( target=start_websocket, args=(action_queue, confidence_queue, pose_queue))
    p_stream = Process( target=start_stream, args=())
    p_websocket.start()
    p_stream.start()
