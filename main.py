from fastapi import FastAPI,Request,Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

# -----------------------------------------------
#  Setup FastAPI
# -----------------------------------------------
app = FastAPI()  
templates=Jinja2Templates(directory="templates")

# -----------------------------------------------
#  Predict Page: Objection detection with YOLOv8
# -----------------------------------------------
import numpy as np
import cv2
from ultralytics import YOLO
from fastapi import File,UploadFile
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend

# 输入是numpy格式，输出也是numpy格式
# @app.post("/detect/")
async def yolo_detect_objects(image):

    def preprocess_letterbox(image):
        letterbox = LetterBox(new_shape=640, stride=32, auto=True)
        image = letterbox(image=image)
        image = (image[..., ::-1] / 255.0).astype(np.float32) # BGR to RGB, 0 - 255 to 0.0 - 1.0
        image = image.transpose(2, 0, 1)[None]  # BHWC to BCHW (n, 3, h, w)
        image = torch.from_numpy(image)
        return image

    def preprocess_warpAffine(image, dst_width=640, dst_height=640):
        scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
        ox = (dst_width  - scale * image.shape[1]) / 2
        oy = (dst_height - scale * image.shape[0]) / 2
        M = np.array([
            [scale, 0, ox],
            [0, scale, oy]
        ], dtype=np.float32)
        
        img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        IM = cv2.invertAffineTransform(M)

        img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
        img_pre = img_pre.transpose(2, 0, 1)[None]
        img_pre = torch.from_numpy(img_pre)
        return img_pre, IM

    def iou(box1, box2):
        def area_box(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        left   = max(box1[0], box2[0])
        top    = max(box1[1], box2[1])
        right  = min(box1[2], box2[2])
        bottom = min(box1[3], box2[3])
        cross  = max((right-left), 0) * max((bottom-top), 0)
        union  = area_box(box1) + area_box(box2) - cross
        if cross == 0 or union == 0:
            return 0
        return cross / union

    def NMS(boxes, iou_thres):

        remove_flags = [False] * len(boxes)

        keep_boxes = []
        for i, ibox in enumerate(boxes):
            if remove_flags[i]:
                continue

            keep_boxes.append(ibox)
            for j in range(i + 1, len(boxes)):
                if remove_flags[j]:
                    continue

                jbox = boxes[j]
                if(ibox[5] != jbox[5]):
                    continue
                if iou(ibox, jbox) > iou_thres:
                    remove_flags[j] = True
        return keep_boxes

    def postprocess(pred, IM=[], conf_thres=0.25, iou_thres=0.45):

        # 输入是模型推理的结果，即8400个预测框
        # 1,8400,84 [cx,cy,w,h,class*80]
        boxes = []
        for item in pred[0]:
            cx, cy, w, h = item[:4]
            label = item[4:].argmax()
            confidence = item[4 + label]
            if confidence < conf_thres:
                continue
            left    = cx - w * 0.5
            top     = cy - h * 0.5
            right   = cx + w * 0.5
            bottom  = cy + h * 0.5
            boxes.append([left, top, right, bottom, confidence, label])

        boxes = np.array(boxes)
        lr = boxes[:,[0, 2]]
        tb = boxes[:,[1, 3]]
        boxes[:,[0,2]] = IM[0][0] * lr + IM[0][2]
        boxes[:,[1,3]] = IM[1][1] * tb + IM[1][2]
        boxes = sorted(boxes.tolist(), key=lambda x:x[4], reverse=True)
        
        return NMS(boxes, iou_thres)

    def hsv2bgr(h, s, v):
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        r, g, b = 0, 0, 0

        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        elif h_i == 5:
            r, g, b = v, p, q

        return int(b * 255), int(g * 255), int(r * 255)

    def random_color(id):
        h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
        s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
        return hsv2bgr(h_plane, s_plane, 1)

    # img_pre = preprocess_letterbox(img)
    img_pre, IM = preprocess_warpAffine(image)

    model  = AutoBackend(weights="yolov8s.pt")
    names  = model.names
    result = model(img_pre)[0].transpose(-1, -2)  # 1,8400,84

    boxes  = postprocess(result, IM)

    for obj in boxes:
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])
        color = random_color(label)
        cv2.rectangle(image, (left, top), (right, bottom), color=color ,thickness=2, lineType=cv2.LINE_AA)
        caption = f"{names[label]} {confidence:.2f}"
        w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
        cv2.rectangle(image, (left - 3, top - 33), (left + w + 10, top), color, -1)
        cv2.putText(image, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

    # cv2.imwrite("infer.jpg", image)
    # print("save done")  

    # Convert image to base64 string
    # _, image_encoded = cv2.imencode(".jpg", image)
    # image_base64 = base64.b64encode(image_encoded).decode()

    return image

import dlib
from io import BytesIO
import imageio
import re

def get_frame(video_frame):
    # 从视频帧中获取图像
    image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    
    # 初始化 dlib 的人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 进行人脸检测
    dets = detector(image, 1)
    
    # 绘制检测到的框
    for d in dets:
        cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)
    
    # 将图像转换回 BGR 格式
    result_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return result_frame


# 输入是文件路径
def dlib_detect_objects(video_path):

    print("0*************")
    print(video_path)
    print("1*************")
    if not isinstance(video_path, str):
        raise ValueError("The video_path parameter should be a string representing the file path.")
    cap = cv2.VideoCapture(video_path)
    frames = []
    print("2*************")
    tmp=0
    while cap.isOpened():
        print(tmp)
        tmp+=1
        ret, frame = cap.read()
        if not ret:
            break
        # 调用人脸检测函数处理每一帧
        frame = get_frame(frame)
        frames.append(frame)
    
    cap.release()
    print("3*************")
    # 编码处理后的帧为视频格式
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print("4*************")
    out_path = 'output.mp4'
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height))
    print("5*************")
    for frame in frames:
        out.write(frame)
    print("6*************")
    out.release()
    #TODO: 路径需要修改
    absolute_path='D:/procedures/mdc/'
    full_path = os.path.join(absolute_path, out_path)

    print(full_path)
    return full_path

# gradio 界面
import gradio as gr

with gr.Blocks() as demo1:
    gr.Interface(fn=yolo_detect_objects,inputs=gr.Image(label="Upload the picture"),outputs=gr.Image(),description="Start objection detection with YOLOv8 here")

with gr.Blocks() as demo2:
    gr.Interface(fn=dlib_detect_objects,inputs=gr.Video(),outputs=gr.Video(),title="Live Webcam", description="Start objection detection with dlib here",allow_flagging='never')
    # gr.Interface(fn=yolo_detect_objects,inputs=gr.Image(label="Upload the picture"),outputs=gr.Image(),description="Start objection detection with YOLOv8 here")

app=gr.mount_gradio_app(app,gr.TabbedInterface([demo1,demo2],["YOLOv8","dlib"]),path="/")


if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)