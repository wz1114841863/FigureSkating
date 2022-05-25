import json
json_input = '/home/dell/first/AlphaPose/examples/demo/output/alphapose-results.json'
json_output = '/home/dell/first/AlphaPose/examples/demo/output/refine-results.json'
with open(json_input, 'r') as f:
    json_load = json.load(f)

video_len = int(json_load[-1]['image_id'].split('.')[0]) + 1
video_ins = []
for ins in json_load:
    video_ins.append(int(ins['image_id'].split('.')[0]))

video_count = []
res = []
point = 0

# check
for idx in range(video_len):
    video_count.append(0)
    res.append([])
    while video_ins[point] == idx:
        res[idx].append(json_load[point])
        video_count[-1] = video_count[-1] + 1
        point += 1
        if  point > len(video_ins)-1:
            break
    if video_count[-1] != 2:
        print(str(idx) + ': ' + str(video_count[-1]) + '   point ' +str(point))
    # if video_count[-1] == 0:
    #     print(str(idx) + ': ' + str(video_count[-1]) + '   point ' +str(point))


# idx 最开始一定要只有两个人
# 初始化分配id 0, 1
res_idx = [[0, 1]] * video_len
res_idx[0][0] = res[0][0]['idx']%2
res_idx[0][1] = res[0][1]['idx']%2

for idx in range(1, video_len):

    # 1box -> nbox(n>3), res删除置信度较低的n-2个
    if len(res[idx-1]) == 1 and len(res[idx]) > 2 :
        remove_item = []
        remove_item.append(res[idx][0])
        remove_item.append(res[idx][1])
        for i in range(2, len(res[idx])):
            remove_item.sort(key = lambda x: x['score'])
            if res[idx][i]['score'] < remove_item[0]['score']:
                remove_item[0] = res[idx][i]
        for i in remove_item:
            res[idx].remove(i)
        # for i in range(len(res[idx])):
        #     if res[idx][i]['idx'] not in [res[idx-1][pre_idx]['idx'] for pre_idx in range(len(res[idx-1]))]:
        #         remove_item.append(res[idx][i])
        # for i in remove_item:
        #     res[idx].remove(i)

    if res[idx-1][0]['idx'] == res[idx][0]['idx']:
        res_idx[idx][0] = res_idx[idx-1][0]
        res_idx[idx][1] = 1 - res_idx[idx-1][0]
    else:
        res_idx[idx][0] = 1 - res_idx[idx-1][0]
        res_idx[idx][1] = res_idx[idx-1][0]

for frame_idx in range(len(res)):
    for instance_idx in range(len(res[frame_idx])):
        res[frame_idx][instance_idx]['idx'] = res_idx[frame_idx][instance_idx]

# check
for frame_idx in range(len(res)):
    if len(res[frame_idx]) > 2:
        print("error")

# with open(json_output, 'w') as f:
#     json_load = json.dump(res, f)