from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import threading
import time
import json
import os
from datetime import datetime

app = Flask(__name__)

# 全局变量
model = None
latest_detections = []
frame_lock = threading.Lock()
latest_frame = None
camera = None
is_detecting = False
detection_stats = {
    'total_detections': 0,
    'fps': 0,
    'last_update': time.time()
}

def init_model():
    """初始化YOLO模型"""
    global model
    try:
        # 加载模型（会自动下载如果不存在）
        model = YOLO('yolov8n.pt')
        print("YOLOv8模型加载成功!")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

def generate_frames():
    """生成视频流帧"""
    global camera, latest_frame, latest_detections, is_detecting, detection_stats
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    start_time = time.time()
    
    while is_detecting and camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        
        # 进行目标检测
        if model is not None:
            try:
                results = model(frame, conf=0.5)  # 设置置信度阈值
                annotated_frame = results[0].plot()
                
                # 更新检测结果
                with frame_lock:
                    latest_frame = annotated_frame
                    latest_detections.clear()
                    
                    if results[0].boxes:
                        boxes = results[0].boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = model.names[class_id]
                            
                            # 计算中心点
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            
                            latest_detections.append({
                                'class': class_name,
                                'confidence': round(confidence, 2),
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'center': [cx, cy]
                            })
                
                # 更新统计信息
                frame_count += 1
                if frame_count % 30 == 0:
                    detection_stats['fps'] = frame_count / (time.time() - start_time)
                    frame_count = 0
                    start_time = time.time()
                    detection_stats['total_detections'] = len(latest_detections)
                    detection_stats['last_update'] = time.time()
                
            except Exception as e:
                print(f"检测错误: {e}")
                annotated_frame = frame
        else:
            annotated_frame = frame
        
        # 编码为JPEG格式
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # 清理资源
    if camera:
        camera.release()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """获取检测结果API"""
    return jsonify({
        'detections': latest_detections,
        'stats': detection_stats,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """开始检测"""
    global is_detecting
    if not is_detecting:
        is_detecting = True
        return jsonify({'status': 'started', 'message': '检测已开始'})
    else:
        return jsonify({'status': 'running', 'message': '检测已在运行中'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """停止检测"""
    global is_detecting, camera
    is_detecting = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped', 'message': '检测已停止'})

@app.route('/capture', methods=['POST'])
def capture_image():
    """捕获当前帧"""
    try:
        with frame_lock:
            if latest_frame is not None:
                # 创建保存目录
                os.makedirs('captures', exist_ok=True)
                
                # 生成文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'captures/capture_{timestamp}.jpg'
                
                # 保存图像
                cv2.imwrite(filename, latest_frame)
                
                return jsonify({
                    'status': 'success', 
                    'message': f'图像已保存: {filename}',
                    'filename': filename
                })
            else:
                return jsonify({'status': 'error', 'message': '没有可用的图像'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'保存失败: {str(e)}'})

@app.route('/status')
def get_status():
    """获取系统状态"""
    return jsonify({
        'is_detecting': is_detecting,
        'model_loaded': model is not None,
        'camera_active': camera is not None and camera.isOpened(),
        'stats': detection_stats
    })

# 初始化应用 - 修复版本
with app.app_context():
    is_detecting = True
    if init_model():
        print("YOLOv8模型加载成功!")
    else:
        print("YOLOv8模型加载失败，应用将继续运行但无法进行检测")

if __name__ == '__main__':
    app.run()