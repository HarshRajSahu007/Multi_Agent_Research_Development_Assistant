import os
from pypdf import PdfReader
import cv2
import pytesseract
from PIL import Image

class DocumentProcessor:
    """Basic Document processor that extracts text and images from PDF files"""
    def __init__(self,output_dir="./extracted_data"):
        self.output_dir=output_dir
        os.makedirs(output_dir,exist_ok=True)

    def process_pdf(self,pdf_path):
        """Process a PDF file and extract text and images"""
        print(f"Processing PDF: {pdf_path}")

        result={
            "text":[],
            "images":[],
            "metadata":{}
        }

        try:
            reader= PdfReader(pdf_path)
            result["metadata"]["title"]=os.path.basename(pdf_path)
            result["metadata"]["pages"]=len(reader.pages)

            for i ,page in enumerate(reader.pages):
                text=page.extract_text()
                if text:
                    result["text"].append({
                        "page":i+1,
                        "content":text
                    })
                    print(f"Extracted text from page {i+1}")
            print(f"Successfully processed {pdf_path}")
            return result
        except Exception as e:
            print(f"Error processing {pdf_path}")
            return None
        
    def extract_images(self,pdf_path):
        """Extract images from a PDF file (basic implementation)"""
        print("Image extraction not fully implemented yet")
        return []
    

    def extract_figures_with_cv(self,image_path):
        """Use OpenCV to detect figures in an image"""
        try:
            img=cv2.imread(image_path)
            if img is None:
                print(f"Could not load the image: {image_path}")
                return []
            
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            edges=cv2.Canny(gray,50,150)

            contours, _ =cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            figures=[]

            for i,contour in enumerate(contours):
                x,y,w,h=cv2.boundingRect(contour)

                if w>100 and h>100:
                    figures.append({
                        "id":i,
                        "box":(x,y,w,h),
                        "area":w*h
                    })
            print(f"Detected {len(figures)} potential figures")
            return figures
        
        except Exception as e:
            print(f"Error processing image {image_path}:{str(e)}")
            return []
        
    def ocr_text(self,image_path):
        """Extract text from an image using OCR"""
        try:
            img=Image.open(image_path)
            text=pytesseract.image_to_string(img)
            return text
        except Exception as e:
            print(f"OCR error on {image_path}: {str(e)}")
            return ""