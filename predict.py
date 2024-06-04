from mmpose.apis import MMPoseInferencer

def pose_infe(video_path:str):
    # instantiate the inferencer using the model alias
    inferencer = MMPoseInferencer('wholebody')

    # The MMPoseInferencer API employs a lazy inference approach,
    # creating a prediction generator when given input
    result_generator = inferencer(video_path, pred_out_dir='predictions',out_dir="output")
    results = [result for result in result_generator]

    # print("**************")
    # print(result_generator)
    # print("**************")
    # print(results)
    # print("**************")

pose_infe("video.mp4")