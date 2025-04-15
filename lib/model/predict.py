import torch
from torchvision.ops.boxes import box_iou

def valid(model, dataloader, device):
    model.eval()
    total_iou = 0
    num_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                gt_boxes = target['boxes']
                
                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue  # Nếu không có dự đoán hoặc ground truth, bỏ qua

                ious = box_iou(pred_boxes, gt_boxes)  # Tính IoU giữa các box dự đoán và ground truth
                total_iou += ious.mean().item()  # Lấy trung bình IoU của từng cặp box
                num_samples += 1

    avg_iou = total_iou / num_samples if num_samples > 0 else 0
    return avg_iou  # Trả về IoU trung bình