import os
import time
import cv2
import mediapipe as mp

from flask import Flask, render_template, Response, request, send_from_directory, flash, url_for
from flask import current_app as app
from werkzeug.utils import secure_filename

from src.lstm import ActionClassificationLSTM
from src.video_analyzer import analyse_video, stream_video

app = Flask(__name__)
UPLOAD_FOLDER = './'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load pretrained LSTM model từ file checkpoint
lstm_classifier = ActionClassificationLSTM.load_from_checkpoint("models/saved_model.ckpt")
lstm_classifier.eval()


class DataObject():
    pass

def checkFileType(f: str):
    return f.split('.')[-1] in ['mkv', 'mp4', 'avi', 'mov', 'flv', 'wmv', 'webm']


def cleanString(v: str):
    out_str = v
    delm = ['_', '-', '.']
    for d in delm:
        out_str = out_str.split(d)
        out_str = " ".join(out_str)
    return out_str


# Cập nhật hàm pose_detector với MediaPipe
def pose_detector(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        keypoints_indices = [
            0,
            2,
            5,  
            11, 
            12, 
            13, 
            14, 
            15, 
            16, 
            23, 
            24, 
            25, 
            26, 
            29, 
            30,
            31,
            32
        ]
        
        keypoints = [(results.pose_landmarks.landmark[i].x * frame.shape[1],
                      results.pose_landmarks.landmark[i].y * frame.shape[0]) for i in keypoints_indices]
        return keypoints
    else:
        return []


@app.route('/', methods=['GET'])
def index():
    obj = DataObject
    obj.video = "Untitled video - Made with Clipchamp.mp4"
    return render_template('/index.html', obj=obj)


@app.route('/upload', methods=['POST'])
def upload():
    obj = DataObject
    obj.is_video_display = False
    obj.video = ""
    if request.method == 'POST' and 'video' in request.files:
        video_file = request.files['video']
        if checkFileType(video_file.filename):
            filename = secure_filename(video_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(filepath)
            obj.video = filename
            obj.is_video_display = True
            return render_template('/index.html', obj=obj)
        else:
            msg = f"{video_file.filename} is not a video file" if video_file.filename else "Please select a video file"
            flash(msg)
        return render_template('/index.html', obj=obj)
    return render_template('/index.html', obj=obj)


@app.route('/sample', methods=['POST'])
def sample():
    obj = DataObject
    obj.is_video_display = True
    obj.video = "Untitled video - Made with Clipchamp.mp4"
    return render_template('/index.html', obj=obj)


@app.route('/files/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/analyzed_files/<filename>')
def get_analyzed_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], "res_{}".format(filename), as_attachment=True)


@app.route('/result_video/<filename>')
def get_result_video(filename):
    stream = stream_video("{}res_{}".format(app.config['UPLOAD_FOLDER'], filename))
    return Response(stream, mimetype='multipart/x-mixed-replace; boundary=frame')


# route definition for video upload for analysis
@app.route('/analyze/<filename>')
def analyze(filename):
    # invokes method analyse_video
    return Response(analyse_video(pose_detector, lstm_classifier, filename), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
