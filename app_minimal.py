from flask import Flask, render_template, jsonify, Response
import time
import random
from datetime import datetime

app = Flask(__name__)

# 模拟检测数据（完全去掉 YOLO）
def get_simulated_detections():
    classes = ['person', 'car', 'cat', 'dog', 'chair', 'bottle', 'cell phone']
    detections = []
    
    # 随机生成1-3个检测结果
    for i in range(random.randint(1, 3)):
        detections.append({
            'class': random.choice(classes),
            'confidence': round(random.uniform(0.7, 0.95), 2),
            'bbox': [
                random.randint(50, 200),
                random.randint(50, 150),
                random.randint(300, 500),
                random.randint(200, 350)
            ],
            'center': [random.randint(100, 400), random.randint(100, 250)]
        })
    
    return detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detections')
def detections():
    return jsonify({
        'detections': get_simulated_detections(),
        'stats': {
            'total_detections': len(get_simulated_detections()),
            'fps': round(random.uniform(15, 30), 1),
            'last_update': time.time(),
            'connected_clients': random.randint(1, 5)
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/status')
def status():
    return jsonify({
        'is_detecting': True,
        'model_loaded': True,
        'camera_active': False,
        'stats': {
            'total_detections': random.randint(0, 3),
            'fps': round(random.uniform(15, 25), 1),
            'last_update': time.time(),
            'connected_clients': random.randint(1, 3)
        }
    })

@app.route('/capture', methods=['POST'])
def capture():
    return jsonify({
        'status': 'success',
        'message': '截图功能在演示模式下不可用',
        'filename': 'demo_capture.jpg'
    })

# 模拟视频流
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            # 返回一个简单的空白帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: 0\r\n\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)