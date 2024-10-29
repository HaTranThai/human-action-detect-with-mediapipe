import cv2
import ntpath
from .utils import draw_keypoints
from .lstm import WINDOW_SIZE
import time

import numpy as np
import torch
import torch.nn.functional as F

LABELS = {
    0: "JUMPING",
    1: "JUMPING_JACKS",
    2: "BOXING",
    3: "WAVING_2HANDS",
    4: "WAVING_1HAND",
    5: "CLAPPING_HANDS",
    6: "RUNNING"
}

# how many frames to skip while inferencing
SKIP_FRAME_COUNT = 0

# analyse the video
def analyse_video(pose_detector, lstm_classifier, video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = ntpath.basename(video_path)
    vid_writer = cv2.VideoWriter('res_{}'.format(file_name), fourcc, 30, (width, height))
    counter = 0
    buffer_window = []
    start = time.time()
    label = None

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        img = frame.copy()
        if(counter % (SKIP_FRAME_COUNT + 1) == 0):
            persons = pose_detector(frame)
            draw_keypoints(persons, img)
            features = []
            for i, row in enumerate(persons):
                features.append(row[0])
                features.append(row[1])

            if len(buffer_window) < WINDOW_SIZE:
                buffer_window.append(features)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32))
                model_input = torch.unsqueeze(model_input, dim=0)
                model_input = model_input.to(device)
                y_pred = lstm_classifier(model_input)
                prob = F.softmax(y_pred, dim=1)
                pred_index = prob.data.max(dim=1)[1]
                buffer_window.pop(0)
                buffer_window.append(features)
                label = LABELS[pred_index.cpu().numpy()[0]]

        if label is not None:
            cv2.putText(img, 'Action: {}'.format(label), (int(width-400), height-50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)
        
        counter += 1
        vid_writer.write(img)
        percentage = int(counter * 100 / tot_frames)
        yield "data:" + str(percentage) + "\n\n"

    analyze_done = time.time()
    print("Video processing finished in ", analyze_done - start)

def stream_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("fps ", fps)
    print("width height", width, height)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("tot_frames", tot_frames)
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        out_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + out_frame + b'\r\n')
        yield result
    print("finished video streaming")
