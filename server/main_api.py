from typing import List
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import time
from queue import Queue

app = FastAPI()


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
    print('write task:{}'.format(index))
    q.put(str(result[index]))
    print( 'write task:{} done'.format( index ) )

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

pose_result = {"left_shoulder": [i for i in range(0,14)],
               "right_shoulder": [i for i in range(10,24)],
               "left_hip": [i for i in range(20,34)],
               "right_hip": [i for i in range(30,44)],
               "left_knee": [i for i in range(40,54)],
               "right_knee": [i for i in range(50,64)],
               "left_ankle": [i for i in range(60,74)],
               "right_ankle": [i for i in range(70,84)]
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
    

# 动作置信度视图，前端用于接收后端发送的动作识别数据，每接收一次，需要更新一次视图中的动作分类结果。
@app.websocket("/ws/action_confidence/result_stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    user = 'confidence'
    try:
        for i in range(len(confidence_result)):
            print(confidence_result[i])
            await asyncio.sleep(1)
            #data = await websocket.receive_text()
            data = str(confidence_result[i])
            await manager.send_personal_message(data, websocket)
            #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端action_confidence断开")
        
# 动作置信度视图，前端用于接收后端发送的动作识别数据，每接收一次，需要更新一次视图中的动作分类结果。
@app.websocket("/ws/action_classification/result_stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    user = 'classification'
    try:
        for i in range(len(action_result)):
            print(action_result[i])
            await asyncio.sleep(1)
            #data = await websocket.receive_text()
            data = str(action_result[i])
            await manager.send_personal_message(data, websocket)
            #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端action_classification断开")
        
# 动作置信度视图，前端用于接收后端发送的动作识别数据，每接收一次，需要更新一次视图中的动作分类结果。
@app.websocket("/ws/pose/result_stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    #await manager.broadcast(f"websocket 后端已连接.")
    user = 'classification'
    try:
        for i in range(len(temp)):
            print(temp[i])
            await asyncio.sleep(1)
            #data = await websocket.receive_text()
            data = str(temp[i])
            await manager.send_personal_message(data, websocket)
            #await manager.broadcast("返回置信度数据:0,1,2,3,4,5")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"终端pose断开")
        
def start_process(action_queue, confidence_queue, pose_queue):
    import uvicorn
    # 官方推荐是用命令后启动 uvicorn main:app --host=127.0.0.1 --port=8010 --reload
    uvicorn.run(app='main_api:app', host="10.112.6.220", port=4011, reload=True, debug=True)
        


if __name__ == "__main__":
    import uvicorn
    # 官方推荐是用命令后启动 uvicorn main:app --host=127.0.0.1 --port=8010 --reload
    uvicorn.run(app='main_api:app', host="10.112.6.220", port=4011, reload=True, debug=True)
