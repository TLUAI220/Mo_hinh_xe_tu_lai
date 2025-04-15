import torch
from torchvision.ops.boxes import box_iou
def train_and_evaluate(model, 
                       train_loader,
                       test_loader,  
                       optimizer, 
                       num_epochs=10, 
                       device="cpu"):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        count = 1
        model.train()
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            count += 1 
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}| Loss: {epoch_loss:.4f}| AVG_Loss {avg_loss}")
        
        model.eval()  
        total_iou = 0
        num_samples = 0

        with torch.no_grad():
            for images, targets in test_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                for output, target in zip(outputs, targets):
                    pred_boxes = output['boxes']
                    gt_boxes = target['boxes']
                    
                    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                        continue

                    ious = box_iou(pred_boxes, gt_boxes)
                    total_iou += ious.mean().item()
                    num_samples += 1

        avg_iou = total_iou / num_samples if num_samples > 0 else 0
        print(f"Test IoU after Epoch {epoch+1}: {avg_iou:.4f}")
    
    return model