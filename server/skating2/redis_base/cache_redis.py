import redis

# action_queue ---- db0
# pose_queue ---- db1
# confidence_queue ---- db2

# main_video_stream ---- db3
# heatmap_video_stream ---- db4
# 3d_video_stream ---- db5

redis_keys = ['main_video_stream', 'heatmap_video_stream', '3d_video_stream', 
              'confidence_queue', 'action_queue', 'pose_queue', 'input_3d']


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
    

def get_frame():
    import cv2
    cap1 = cv2.VideoCapture(r"/home/dell/wz/AlphaPose/server/peng_jin-2022-01-17.mp4")

    res,frame = cap1.read()
    ret, jpeg = cv2.imencode('.jpg', frame)
    wirte_into_queue(redis_conn, 'main_video_stream', jpeg.tobytes())
    jpeg = eval(get_from_queue(redis_conn, 'main_video_stream'))


redis_pool = redis.ConnectionPool(host='127.0.0.1', port= 6379, password= '159753', db= 0)
redis_conn = redis.Redis(connection_pool= redis_pool)

def delete_key_redis(redis_conn, key):
    redis_conn.delete(key)
    
def init_redis(redis_conn):
    for key in redis_keys:
        delete_key_redis(redis_conn, key)

def list_is_empty(r, key):
    """judge wheather the list is empty.

    Args:
        r (redis.Redis): the connection to redis.
        key (_type_): the name of the list.

    Returns:
        bool: True--empty; False--NOT empty.
    """
    #list1 = r.lrange(key , 0, -1)   # 0到-1表示 取所有元素
    # 如果key1不存在的话， list1 = []
    list1 = r.llen(key)
    if list1 <= 0:
        #print('key不存在！')
        return True
    else:
        #print('key存在！')
        return False

def wirte_into_queue(redis_conn, key, result,is_byte = False):
    """write the result into the redis[key].

    Args:
        redis_conn (redis.Redis): the connection to redis.
        key (str): the name of the list.
        result (stringable): the object which will be write into the redis. 
    """
    if is_byte == False:
        v = redis_conn.lpush(key, str(result))
    else:
        v = redis_conn.lpush(key, result)
    
def get_from_queue(redis_conn, key):
    """get value from redis list.

    Args:
        redis_conn (redis.Redis): the connection to redis.
        key (str): the name of the list.

    Returns:
        bytes : the value of the list which is popped. could be .decode('utf-8') to string.
    """
    v = redis_conn.rpop(key)
    #print(v.decode('utf-8'))
    return v
    
if __name__ == "__main__":
    key = 'main_video_stream'
    import pdb;pdb.set_trace()
    get_frame()
    list_is_empty(redis_conn, key)
    wirte_into_queue(redis_conn, key, pose_result)
    wirte_into_queue(redis_conn, key, action_result)
    list_is_empty(redis_conn, key)
    get_from_queue(redis_conn, key)
    list_is_empty(redis_conn, key)
    get_from_queue(redis_conn, key)
    list_is_empty(redis_conn, key)
