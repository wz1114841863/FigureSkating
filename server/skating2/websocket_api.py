from lib2to3.pytree import convert
from typing import List
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import time
from queue import Queue
import redis
from redis_base.cache_redis import wirte_into_queue, get_from_queue, list_is_empty, redis_pool
redis_conn = redis.Redis(connection_pool= redis_pool)
app = FastAPI()

ensumble = False

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

# 写数据进程执行的代码
def write(q, result, index = None):
    #print('write task:{}'.format(index))
    q.put(str(result[index]))
    #print( 'write task:{} done'.format( index ) )


action_list = { 0:"CSp",
               60:"CSp",
               120:"USp",
               180:"USp",
               240:"SSp",
               300:"SSp"}

action = {"0": ["CSp", ["2.2057588e-05", "7.490687e-05", "6.517914e-06", "0.9938122", "0.0029665744", "0.0001335177", "4.4689757e-05", "1.8377577e-05", "1.1723739e-05", "0.0028494052", "6.0010436e-05"]], 
          "60": ["CSp", ["1.1274169e-05", "5.3218653e-05", "3.0850442e-06", "0.9805232", "0.0032843538", "0.00023912531", "1.7318958e-05", "1.7441911e-05", "8.748565e-06", "0.015654275", "0.00018796859"]], 
          "120": ["USp", ["1.3451221e-06", "2.0268008e-06", "5.2943703e-09", "0.00028206361", "4.0322237e-05", "0.7528302", "3.6845697e-07", "9.742555e-07", "1.0118602e-07", "0.24683198", "1.0629152e-05"]], 
          "180": ["USp", ["8.9978937e-07", "1.0366833e-06", "1.2891178e-08", "1.38439955e-05", "1.3923961e-06", "0.99984425", "2.4667858e-07", "1.4759928e-07", "1.7857552e-08", "0.00012246182", "1.5815056e-05"]], 
          "240": ["SSp", ["6.844883e-05", "8.01473e-06", "7.011096e-06", "3.1303694e-06", "0.98856914", "0.0010324251", "3.3245713e-06", "1.174061e-06", "9.3655717e-07", "0.010305816", "6.3578614e-07"]], 
          "300": ["SSp", ["9.328187e-07", "2.4741044e-07", "1.8147755e-07", "5.623458e-06", "0.90970683", "0.0009966717", "2.064778e-07", "5.4034615e-08", "3.9139074e-08", "0.08928913", "1.2218935e-07"]]}

action_map = {
    "T" : '后外点冰跳',
    "F"  : '后内点冰跳',
    "Lz" : '勾手跳',
    "S" : '后内结环跳',
    "Lo" : '后外结环跳',
    "A" : '阿克塞尔跳',
    "CSp": "燕式旋转",
    "USp" : '直立旋转',
    "SSp" : '蹲踞旋转',
    "NB" : '非基本姿态旋转'
}

labels = ['T', 'Lo', 'A', 'CSp', 'SSp', 'USp', 'S', 'F', 'Lz', 'NB', 'Null']

def convert_order(confidence_list):
    show_order = [0,7,8,6,1,2,3,5,4,9]
    temp = []
    for i in range(10):
        temp.append(float(confidence_list[show_order[i]]))
    return temp

action_result = {}
confidence_result = {}
action_key = list(action.keys())
for i in range(len(action)):
    #action_result[int(action_key[i])] = action_map[action[action_key[i]][0]]
    action_result[int(action_key[i])] = action_map[action[action_key[i]][0]]
    confidence_result[int(action_key[i])] = action[action_key[i]][1]
    for j in range(len(confidence_result[int(action_key[i])])):
        confidence_result[int(action_key[i])][j] = float(confidence_result[int(action_key[i])][j])


temp = {0: 
            ["2.2057588e-05", "1.8377577e-05", "0.0028494052", "0.0001335177", "7.490687e-05", "6.0010436e-05", "4.4689757e-05", "0.9938122", "0.0029665744", "1.1723739e-05", "6.517914e-06"]}



import random

pose_result = {"left_shoulder": [random.randint(20,90) for i in range(0,14)],
               "right_shoulder": [random.randint(20,90) for i in range(10,24)],
               "left_hip": [random.randint(20,90) for i in range(20,34)],
               "right_hip": [random.randint(20,90) for i in range(30,44)],
               "left_knee": [random.randint(20,90) for i in range(40,54)],
               "right_knee": [random.randint(20,90) for i in range(50,64)],
               "left_ankle": [random.randint(20,90) for i in range(60,74)],
               "right_ankle": [random.randint(20,90) for i in range(70,84)]
               }
temp = []
for i in range(len(pose_result['left_shoulder'])):
    temp.append({"left_shoulder": pose_result['left_shoulder'][i],
               "right_shoulder": pose_result['right_shoulder'][i],
               "left_hip": pose_result['left_hip'][i],
               "right_hip": pose_result['right_hip'][i],
               "left_knee": pose_result['left_knee'][i],
               "right_knee": pose_result['right_knee'][i],
               "left_ankle": pose_result['left_ankle'][i],
               "right_ankle": pose_result['right_ankle'][i]
               })
    

def init_redis_websocket(redis_conn, key):
    if key == 'confidence_queue':
        for i in range(6):
            wirte_into_queue(redis_conn, 'confidence_queue', str({list(action_list.keys())[i]:confidence_result[i*60]}))
    elif key == 'action_queue':
        for i in range(6):
            wirte_into_queue(redis_conn, 'action_queue', str({list(action_list.keys())[i]:action_list[i*60]}))
    elif key == 'pose_queue':
        for i in range(14):
            wirte_into_queue(redis_conn, 'pose_queue', str(temp[i]))
        
def delete_key_redis(redis_conn, key):
    redis_conn.delete(key)

#----------------------------------------------------------
@app.websocket("/ws/action_confidence/result_stream_test")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    key = 'confidence_queue'
    delete_key_redis(redis_conn, key)
    init_redis_websocket(redis_conn, key)
    cn = 0 
    interval_t = 0
    interval_pre = -60
    consume_time_start = 0
    consume_time_end = 0
    try:
        while True:
            if not list_is_empty(redis_conn, key):
                interval_consume = consume_time_end - consume_time_start
                interval_t = interval_t - (interval_consume) if interval_t - (interval_consume) <= interval_t and interval_t - (interval_consume) > 0 else 0
                await asyncio.sleep(interval_t)
                consume_time_start = time.time()
                #data = await websocket.receive_text()
                data = eval(get_from_queue(redis_conn, key).decode('utf-8'))
                interval_post, data = list(data.keys())[0], data[list(data.keys())[0]]
                data = convert_order(data)
                interval_t = float((interval_post - interval_pre)/25)
                interval_pre = interval_post
                await manager.send_personal_message(str(data), websocket)
                #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")
                consume_time_end = time.time()
            else:
                await asyncio.sleep(2.4)
                print(key, " is empty.")
                cn = cn + 1
                if cn == 2: break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端action_confidence断开")


@app.websocket("/ws/action_classification/result_stream_test")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    cn = 0
    key = 'action_queue'
    delete_key_redis(redis_conn, key)
    init_redis_websocket(redis_conn, key)
    interval_t = 0
    interval_pre = -60
    consume_time_start = 0
    consume_time_end = 0
    try:
        while True:
            if not list_is_empty(redis_conn, key):
                interval_consume = consume_time_end - consume_time_start
                interval_t = interval_t - (interval_consume) if interval_t - (interval_consume) <= interval_t and interval_t - (interval_consume) > 0 else 0
                await asyncio.sleep(interval_t)
                print(interval_t)
                consume_time_start = time.time()
                #data = await websocket.receive_text()
                data = eval(get_from_queue(redis_conn, key).decode('utf-8'))
                interval_post, data = list(data.keys())[0], data[list(data.keys())[0]]
                interval_t = float((interval_post - interval_pre)/25)
                interval_pre = interval_post
                await manager.send_personal_message(action_map[data], websocket)
                consume_time_end = time.time()
                #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")
            else:
                await asyncio.sleep(interval_t)
                print(key, " is empty.")
                cn = cn + 1
                if cn == 2: break
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端action_classification断开")
        
@app.websocket("/ws/pose/result_stream_test")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    key = 'pose_queue'
    cn = 0
    delete_key_redis(redis_conn, key)
    init_redis_websocket(redis_conn, key)
    try:
        while  True:
            if not list_is_empty(redis_conn, key):
                await asyncio.sleep(1)
                data = get_from_queue(redis_conn, key).decode('utf-8')
                await manager.send_personal_message(data, websocket)

            else:
                await asyncio.sleep(1)
                print(key, " is empty.")
                cn = cn + 1
                if cn == 2: break
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端pose断开")


#----------------------------------------------------------------------
@app.websocket("/ws/action_confidence/result_stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    key = 'confidence_queue'
    cn = 0 
    interval_t = 0
    interval_pre = -60
    consume_time_start = 0
    consume_time_end = 0
    try:
        while True:
            if not list_is_empty(redis_conn, key):
                interval_consume = consume_time_end - consume_time_start
                interval_t = interval_t - (interval_consume) if interval_t - (interval_consume) <= interval_t and interval_t - (interval_consume) > 0 else 0
                await asyncio.sleep(interval_t)
                consume_time_start = time.time()
                #data = await websocket.receive_text()
                data = eval(get_from_queue(redis_conn, key).decode('utf-8'))
                interval_post, data = list(data.keys())[0], data[list(data.keys())[0]]
                data = convert_order(data)
                interval_t = float((interval_post - interval_pre)/25)
                interval_pre = interval_post
                await manager.send_personal_message(str(data), websocket)
                #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")
                consume_time_end = time.time()
            else:
                await asyncio.sleep(2.4)
                print(key, " is empty.")
                cn = cn + 1
                if cn == 2: break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端action_confidence断开")


@app.websocket("/ws/action_classification/result_stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    cn = 0
    key = 'action_queue'
    interval_t = 0
    interval_pre = -60
    consume_time_start = 0
    consume_time_end = 0
    try:
        while True:
            if not list_is_empty(redis_conn, key):
                interval_consume = consume_time_end - consume_time_start
                interval_t = interval_t - (interval_consume) if interval_t - (interval_consume) <= interval_t and interval_t - (interval_consume) > 0 else 0
                await asyncio.sleep(interval_t)
                print(interval_t)
                consume_time_start = time.time()
                #data = await websocket.receive_text()
                data = eval(get_from_queue(redis_conn, key).decode('utf-8'))
                interval_post, data = list(data.keys())[0], data[list(data.keys())[0]]
                interval_t = float((interval_post - interval_pre)/25)
                interval_pre = interval_post
                await manager.send_personal_message(action_map[data], websocket)
                consume_time_end = time.time()
                #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")
            else:
                await asyncio.sleep(interval_t)
                print(key, " is empty.")
                cn = cn + 1
                if cn == 2: break
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端action_classification断开")
        
@app.websocket("/ws/pose/result_stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    key = 'pose_queue'
    cn = 0
    try:
        while  True:
            if not list_is_empty(redis_conn, key):
                await asyncio.sleep(1)
                #data = await websocket.receive_text()
                data = get_from_queue(redis_conn, key).decode('utf-8')
                await manager.send_personal_message(data, websocket)
                #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")
            else:
                await asyncio.sleep(1)
                print(key, " is empty.")
                cn = cn + 1
                if cn == 2: break
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端pose断开")
        

        
def start_websocket(action_queue, confidence_queue, pose_queue):
    import uvicorn
    # 官方推荐是用命令后启动 uvicorn main:app --host=127.0.0.1 --port=8010 --reload
    uvicorn.run(app='websocket_api:app', host="10.112.6.220", port=4011, reload=True, debug=True)
        


if __name__ == "__main__":
    import uvicorn
    # 官方推荐是用命令后启动 uvicorn main:app --host=127.0.0.1 --port=8010 --reload
    uvicorn.run(app='websocket_api:app', host="10.112.6.220", port=4011, reload=True, debug=True)
