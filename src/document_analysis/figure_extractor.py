import torch
from torchvision.models import detection
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import cv2

class FigureExtractor:
    """Extract Figures from document using PyTorch vision models."""
    def __init__(self, confidence_threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device : {self.device}")

        self.model=detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        self.confidence_threshold=confidence_threshold

        self.classes = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
            'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
            'toothbrush'
        ]

    def detect_objects(self,image_path):
        """Detect Objects in an image using Faster R-CNN."""
        try:
            image= Image.open(image_path).convert('RGB')
            image_tensor=F.to_tensor(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions= self.model(image_tensor)

            results=[]
            pred_boxes=predictions[0]['boxes'].cpu().numpy()
            pred_scores=predictions[0]['scores'].cpu().numpy()
            pred_labels=predictions[0]['labels'].cpu().numpy()

            for box,score,label in zip(pred_boxes,pred_scores,pred_labels):
                if score >=self.confidence_threshold:
                    x1,y1,x2,y2= box.astype(int)
                    class_name=self.classes[label]

                    results.append({
                        'box':(x1,y1,x2,y2),
                        'width': x2-x1,
                        'height':y2-y1,
                        'score':float(score),
                        'class':class_name
                    })

            print(f"Detected {len(results)} objects with confidence>= {self.confidence_threshold}")
            return results
        
        except Exception as e:
            print(f"Error detection objects in {image_path}: {str(e)}")
            return []
        

    def extract_diagram_regions(self,image_path):
        """
        Detect potential diagram regions in scientific papers.
        This is a starting point - a specialized model would be better
        """
        objects=self.detect_objects(image_path)

        image=Image.open(image_path)

        width,height=image.size

        diagrams=[]

        object_regions=[]

        for obj in objects:
            if obj['class'] in ['book','tv','laptop','cell phone']:
                object_regions.append(obj)

        try:
            img_np=np.array(image.convert('RGB'))
            img_cv=img_np[:,:,::-1].copy()

            gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Consider regions that are large enough and have reasonable aspect ratio
                area = w * h
                aspect_ratio = max(w, h) / min(w, h)
                
                if area > 0.05 * width * height and aspect_ratio < 5:
                    diagrams.append({
                        'box': (x, y, x+w, y+h),
                        'width': w,
                        'height': h,
                        'area': area,
                        'type': 'potential_diagram'
                    })
        except Exception as e:
            print(f"Error in diagram detection: {str(e)}")
        

        all_regions = object_regions + diagrams
        print(f"Found {len(all_regions)} potential diagram regions")
        return all_regions


