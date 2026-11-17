import os
import json
import gc

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & TỐI ƯU BỘ NHỚ CHO SERVER (RENDER)
# ==============================================================================
# Ép thư mục cấu hình tạm của Ultralytics vào /tmp để tránh lỗi phân quyền ghi trên Linux
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp')

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# Giới hạn số luồng CPU của PyTorch về 1 để tiết kiệm tối đa RAM trên gói Free (512MB)
torch.set_num_threads(1)

app = Flask(__name__)
# Giới hạn dung lượng ảnh tải lên tối đa 20 MB
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
# Hiển thị tiếng Việt rõ ràng trong dữ liệu JSON trả về
app.config['JSON_AS_ASCII'] = False
try:
    app.json.ensure_ascii = False
except AttributeError:
    pass

# Danh sách các loại cây trồng hỗ trợ chẩn đoán
CROPS_INFO = {
    'ca-chua': {'name': 'Cà Chua', 'icon': '🍅', 'badge': 'bg-danger'},
    'ca-rot': {'name': 'Cà Rốt', 'icon': '🥕', 'badge': 'bg-warning text-dark'}
}

# ==============================================================================
# 2. QUẢN LÝ ĐƯỜNG DẪN MÔ HÌNH ĐÃ HUẤN LUYỆN
# ==============================================================================
def get_model_paths(crop_type):
    """
    Trả về đường dẫn các file trọng số AI đã huấn luyện sẵn trong thư mục models/.
    """
    models_dir = os.path.abspath("models")
    os.makedirs(models_dir, exist_ok=True)

    mobilenet_path = os.path.join(models_dir, f"{crop_type}_mobilenet.pth")
    yolo_path = os.path.join(models_dir, f"{crop_type}_yolo.pt")
    classes_path = os.path.join(models_dir, f"{crop_type}_classes.json")

    return mobilenet_path, yolo_path, classes_path

# ==============================================================================
# 3. TIỀN XỬ LÝ ẢNH & KHỞI TẠO KIẾN TRÚC MOBILENETV2
# ==============================================================================
# [THẦY HỎI]: Tại sao phải Resize(224, 224) và Normalize theo các thông số bên dưới?
# [TRẢ LỜI]:
# - Resize(224, 224): Đưa ảnh về đúng kích thước đầu vào chuẩn của mạng MobileNetV2.
# - Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]): Chuẩn hóa phân phối màu RGB
#   theo giá trị Mean và Std của tập ImageNet để đồng nhất với lúc huấn luyện.
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def build_mobilenet_model(num_classes):
    """
    [THẦY HỎI]: Tại sao đặt weights=None khi khởi tạo MobileNetV2 ở bước dự đoán?
    [TRẢ LỜI]: Đặt weights=None (pretrained=False) để bỏ qua việc tải trọng số gốc ImageNet
    từ Internet, giúp tiết kiệm RAM và nạp trực tiếp file trọng số (.pth) đã huấn luyện.
    """
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# ==============================================================================
# 4. ROUTE ĐIỀU HƯỚNG GIAO DIỆN WEB
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
# 5. API CHẨN ĐOÁN BỆNH LÁ CÂY (PREDICT)
# ==============================================================================
@app.route('/api/<crop_type>/predict', methods=['POST'])
def predict(crop_type):
    """
    Nhận ảnh từ người dùng, đưa qua các mô hình AI đã huấn luyện (YOLOv8 & MobileNetV2)
    và tổng hợp kết quả chẩn đoán có độ tin cậy cao nhất.
    """
    if crop_type not in CROPS_INFO:
        return jsonify({'error': 'Loại cây không hợp lệ!'}), 400

    mobilenet_path, yolo_path, classes_path = get_model_paths(crop_type)
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Vui lòng chọn hoặc chụp ảnh lá cây để chẩn đoán!'}), 400

    has_mobilenet = os.path.exists(mobilenet_path) and os.path.exists(classes_path)
    has_yolo = os.path.exists(yolo_path)

    if not has_mobilenet and not has_yolo:
        return jsonify({'error': f'Chưa tìm thấy file mô hình AI cho {CROPS_INFO[crop_type]["name"]} trong thư mục models/!'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        # Thu nhỏ ảnh chụp độ phân giải cao từ điện thoại về 224x224 ngay từ đầu
        # Giúp xử lý cực nhanh (~1-2 giây) và chống tràn RAM / lỗi 504 trên server
        image = image.resize((224, 224))
    except Exception:
        return jsonify({'error': 'File tải lên không phải hình ảnh hợp lệ!'}), 400

    results_detail = {}

    # 1. Dự đoán bằng YOLOv8
    if has_yolo:
        try:
            y_model = YOLO(yolo_path)
            y_res = y_model(image, imgsz=224, verbose=False)[0]
            top1_idx = int(y_res.probs.top1)
            y_disease = y_res.names[top1_idx]
            y_conf = round(float(y_res.probs.top1conf.item()) * 100, 2)
            results_detail['yolov8'] = {'disease': y_disease, 'confidence': y_conf}

            # Giải phóng RAM ngay sau khi YOLOv8 chạy xong
            del y_model, y_res
            gc.collect()
        except Exception as e:
            results_detail['yolov8_error'] = str(e)

    # 2. Dự đoán bằng MobileNetV2
    if has_mobilenet:
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = json.load(f)

            mb_model = build_mobilenet_model(len(class_names))
            mb_model.load_state_dict(torch.load(mobilenet_path, map_location=torch.device('cpu')))
            mb_model.eval()

            tensor = eval_transforms(image).unsqueeze(0)
            with torch.no_grad():
                outputs = mb_model(tensor)
                # Dùng hàm Softmax chuyển đổi đầu ra thành tỷ lệ % xác suất
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, idx = torch.max(probs, 0)

            mb_disease = class_names[idx.item()]
            mb_conf = round(conf.item() * 100, 2)
            results_detail['mobilenet'] = {'disease': mb_disease, 'confidence': mb_conf}

            # Giải phóng RAM ngay sau khi MobileNetV2 chạy xong
            del mb_model, tensor, outputs, probs
            gc.collect()
        except Exception as e:
            results_detail['mobilenet_error'] = str(e)

    # 3. Tổng hợp kết quả chẩn đoán giữa 2 mô hình
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
