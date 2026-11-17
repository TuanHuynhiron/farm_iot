import os
import json
import gc
import ctypes

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & TỐI ƯU BỘ NHỚ TỐI ĐA CHO RENDER FREE (512 MB RAM)
# ==============================================================================
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# Tắt hoàn toàn tính toán đạo hàm (Autograd) trên toàn hệ thống và giới hạn 1 luồng CPU
torch.set_grad_enabled(False)
torch.set_num_threads(1)

def trim_memory():
    """Ép hệ điều hành Linux (Render) thu hồi ngay lập tức vùng nhớ RAM trống"""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False
try:
    app.json.ensure_ascii = False
except AttributeError:
    pass

CROPS_INFO = {
    'ca-chua': {'name': 'Cà Chua', 'icon': '🍅', 'badge': 'bg-danger'},
    'ca-rot': {'name': 'Cà Rốt', 'icon': '🥕', 'badge': 'bg-warning text-dark'}
}

# Chỉ giữ mô hình của 1 loại cây đang dùng trong RAM để không bao giờ tràn 512 MB
ACTIVE_CACHE = {'crop_type': None, 'yolo': None, 'mobilenet': None, 'classes': None}

# ==============================================================================
# 2. QUẢN LÝ ĐƯỜNG DẪN & NẠP MÔ HÌNH MỘT LẦN (TRÁNH PHÂN MẢNH RAM)
# ==============================================================================
def get_model_paths(crop_type):
    models_dir = os.path.abspath("models")
    os.makedirs(models_dir, exist_ok=True)

    mobilenet_path = os.path.join(models_dir, f"{crop_type}_mobilenet.pth")
    yolo_path = os.path.join(models_dir, f"{crop_type}_yolo.pt")
    classes_path = os.path.join(models_dir, f"{crop_type}_classes.json")

    return mobilenet_path, yolo_path, classes_path

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def build_mobilenet_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def get_loaded_models(crop_type):
    """
    Nạp sẵn YOLOv8 và MobileNetV2 1 lần cho loại cây đang chọn.
    Nếu đổi sang loại cây khác, giải phóng mô hình cũ trước khi nạp mô hình mới.
    """
    if ACTIVE_CACHE['crop_type'] == crop_type:
        return ACTIVE_CACHE

    ACTIVE_CACHE['crop_type'] = None
    ACTIVE_CACHE['yolo'] = None
    ACTIVE_CACHE['mobilenet'] = None
    ACTIVE_CACHE['classes'] = None
    trim_memory()

    mobilenet_path, yolo_path, classes_path = get_model_paths(crop_type)

    if os.path.exists(yolo_path):
        ACTIVE_CACHE['yolo'] = YOLO(yolo_path)

    if os.path.exists(mobilenet_path) and os.path.exists(classes_path):
        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        mb_model = build_mobilenet_model(len(class_names))
        mb_model.load_state_dict(torch.load(mobilenet_path, map_location=torch.device('cpu')))
        mb_model.eval()
        ACTIVE_CACHE['mobilenet'] = mb_model
        ACTIVE_CACHE['classes'] = class_names

    ACTIVE_CACHE['crop_type'] = crop_type
    trim_memory()
    return ACTIVE_CACHE

# ==============================================================================
# 3. ROUTE ĐIỀU HƯỚNG GIAO DIỆN WEB
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html', crops=CROPS_INFO)

@app.route('/plant/<crop_type>')
def plant_page(crop_type):
    if crop_type not in CROPS_INFO:
        return "Loại cây không hợp lệ!", 404

    mobilenet_path, yolo_path, classes_path = get_model_paths(crop_type)
    has_mobilenet = os.path.exists(mobilenet_path) and os.path.exists(classes_path)
    has_yolo = os.path.exists(yolo_path)

    return render_template(
        'plant.html',
        crop_type=crop_type,
        crop=CROPS_INFO[crop_type],
        has_mobilenet=has_mobilenet,
        has_yolo=has_yolo
    )

# ==============================================================================
# 4. API CHẨN ĐOÁN BỆNH LÁ CÂY (PREDICT)
# ==============================================================================
@app.route('/api/<crop_type>/predict', methods=['POST'])
def predict(crop_type):
    if crop_type not in CROPS_INFO:
        return jsonify({'error': 'Loại cây không hợp lệ!'}), 400

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Vui lòng chọn hoặc chụp ảnh lá cây để chẩn đoán!'}), 400

    models_bundle = get_loaded_models(crop_type)
    y_model = models_bundle['yolo']
    mb_model = models_bundle['mobilenet']
    class_names = models_bundle['classes']

    if y_model is None and mb_model is None:
        return jsonify({'error': f'Chưa tìm thấy file mô hình AI cho {CROPS_INFO[crop_type]["name"]} trong thư mục models/!'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
    except Exception:
        return jsonify({'error': 'File tải lên không phải hình ảnh hợp lệ!'}), 400

    results_detail = {}

    # 1. Dự đoán bằng YOLOv8 (dùng mô hình đã nạp sẵn)
    if y_model is not None:
        try:
            y_res = y_model.predict(image, imgsz=224, verbose=False)[0]
            top1_idx = int(y_res.probs.top1)
            y_disease = y_res.names[top1_idx]
            y_conf = round(float(y_res.probs.top1conf.item()) * 100, 2)
            results_detail['yolov8'] = {'disease': y_disease, 'confidence': y_conf}
            del y_res
        except Exception as e:
            results_detail['yolov8_error'] = str(e)

    # 2. Dự đoán bằng MobileNetV2 (dùng mô hình đã nạp sẵn)
    if mb_model is not None and class_names is not None:
        try:
            tensor = eval_transforms(image).unsqueeze(0)
            with torch.inference_mode():
                outputs = mb_model(tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, idx = torch.max(probs, 0)

            mb_disease = class_names[idx.item()]
            mb_conf = round(conf.item() * 100, 2)
            results_detail['mobilenet'] = {'disease': mb_disease, 'confidence': mb_conf}
            del tensor, outputs, probs
        except Exception as e:
            results_detail['mobilenet_error'] = str(e)

    del image
    trim_memory()

    # 3. Tổng hợp kết quả chẩn đoán
    final_disease, final_confidence = "", 0.0

    if 'yolov8' in results_detail and 'mobilenet' in results_detail:
        y_data, m_data = results_detail['yolov8'], results_detail['mobilenet']
        if y_data['disease'] == m_data['disease']:
            final_disease = y_data['disease']
            final_confidence = round((y_data['confidence'] + m_data['confidence']) / 2, 2)
        else:
            if y_data['confidence'] >= m_data['confidence']:
                final_disease, final_confidence = y_data['disease'], y_data['confidence']
            else:
                final_disease, final_confidence = m_data['disease'], m_data['confidence']
    elif 'yolov8' in results_detail:
        final_disease, final_confidence = results_detail['yolov8']['disease'], results_detail['yolov8']['confidence']
    elif 'mobilenet' in results_detail:
        final_disease, final_confidence = results_detail['mobilenet']['disease'], results_detail['mobilenet']['confidence']
    else:
        return jsonify({'error': 'Lỗi khi chạy mô hình AI!', 'details': results_detail}), 500

    return jsonify({
        'disease': final_disease,
        'confidence': final_confidence,
        'details': results_detail
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
