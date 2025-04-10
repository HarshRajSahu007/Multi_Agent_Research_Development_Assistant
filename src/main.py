# main.py
import os
import argparse
from document_analysis.document_processor import DocumentProcessor

def main():
    parser = argparse.ArgumentParser(description="Research Assistant Document Processor")
    parser.add_argument("--pdf", help="Path to PDF file to process")
    parser.add_argument("--output", default="./extracted_data", help="Output directory")
    args = parser.parse_args()
    
    if not args.pdf:
        print("Please provide a PDF file path using --pdf argument")
        return
    
    if not os.path.exists(args.pdf):
        print(f"PDF file not found: {args.pdf}")
        return
        
    processor = DocumentProcessor(output_dir=args.output)
    result = processor.process_pdf(args.pdf)
    
    if result:
        print(f"Successfully processed document")
        print(f"Extracted {len(result['text'])} pages of text")
        
        # Display a sample of the extracted text (first 200 chars)
        if result['text']:
            sample = result['text'][0]['content'][:200] + "..." if len(result['text'][0]['content']) > 200 else result['text'][0]['content']
            print(f"Sample text: {sample}")
    
if __name__ == "__main__":
    main()