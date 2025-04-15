import os
import cv2
import torch
import gdown
import torchvision
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.getcwd()

def load_model():
    if not os.path.exists(f"{ROOT}/fastercnn_model.pth"):
        gdown.download(f"https://drive.google.com/uc?id=1RME3u4gWjRg3uCQb_D8yQaSIDv0t7sFP", f"{ROOT}/fasterrcnn_model.pth", quiet=False)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 6  # Số lượng class (background + object)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Load trọng số đã huấn luyện
    model.load_state_dict(torch.load("fasterrcnn_model.pth", map_location=device))
    model.to(device)  # Chuyển sang chế độ đánh giá

    # Dùng GPU nếu có
    model.eval()
    return model

def process_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    return image, outputs

def draw_boxes(image, outputs, threshold=0.5):
    labels = outputs[0]['labels'].numpy()
    boxes = outputs[0]['boxes'].numpy()
    scores = outputs[0]['scores'].numpy()
    
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f'{label}: {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg"), ("Video files", "*.mp4;*.avi")])
    if not file_path:
        return
    
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image, outputs = process_image(file_path, model)
        image_with_boxes = draw_boxes(image, outputs)
        
        img_tk = ImageTk.PhotoImage(image_with_boxes)
        label.config(image=img_tk)
        label.image = img_tk
    elif file_path.lower().endswith(('.mp4', '.avi')):
        process_video(file_path, model)

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
        
        image_with_boxes = draw_boxes(image, outputs)
        cv2.imshow('Video Processing', np.array(image_with_boxes))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

model = load_model()

root = tk.Tk()
root.title("Faster R-CNN Object Detection")

btn = tk.Button(root, text="Chọn Ảnh/Video", command=open_file)
btn.pack()

label = tk.Label(root)
label.pack()

root.mainloop()