from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from moviepy.editor import VideoFileClip
import os

# -----------------------------------------------
#  Setup FastAPI
# -----------------------------------------------
app = FastAPI()  
# templates=Jinja2Templates(directory="templates")

# -----------------------------------------------
#  Predict Page: Objection detection with YOLOv8
# -----------------------------------------------
from predict import pose_infe

def mmpose_prediction(video_path):
    pose_infe(video_path)
    file_name=os.path.basename(video_path)
    input_path=os.path.join("D:\\procedures\\mdc\\output\\visualizations",file_name)
    base_name,ext=os.path.splitext(file_name)
    new_bn=base_name+"1"
    new_fn=new_bn+ext
    output_path=os.path.join("D:\\procedures\\mdc\\output\\visualizations",new_fn)
    # path='D:\\procedures\\mdc\\output\\visualizations\\video.mp4'
    video = VideoFileClip(input_path)
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    video.close()
    return output_path

# gradio 界面
import gradio as gr

theme_blue = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
)

with gr.Blocks(theme=theme_blue) as demo:
    gr.Interface(fn=mmpose_prediction,inputs=gr.Video(),outputs=gr.Video(),allow_flagging='never')

app=gr.mount_gradio_app(app,demo,path="/")

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)