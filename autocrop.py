import cv2
import numpy as np
import os
import argparse
from rembg import remove, new_session

# Pre-load session to make processing faster for multiple images
session = None

def get_session():
    global session
    if session is None:
        session = new_session()
    return session

def order_points(pts):
    # Sort points for perspective transform
    # Order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_and_warp(img, box):
    rect = order_points(box)
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def detect_boxes(image_path, min_area_ratio=0.02):
    """
    Uses AI to find image bounding boxes on a scan.
    Returns: (img, list_of_boxes), where a box is an array of 4 points.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    with open(image_path, 'rb') as i:
        input_data = i.read()
        
    output_data = remove(input_data, session=get_session())
    
    nparr = np.frombuffer(output_data, np.uint8)
    out_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if out_img is None or out_img.shape[2] != 4:
        raise ValueError(f"No transparency (AI did not detect background) in file {image_path}")
        
    alpha = out_img[:, :, 3]
    _, thresh = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = img.shape[0] * img.shape[1]
    
    boxes = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) > total_area * min_area_ratio:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            boxes.append(np.int32(box))
            
    return img, boxes

def process_image(image_path, output_dir, min_area_ratio=0.02):
    """
    Function used in CLI mode for automatic cropping and saving.
    """
    print(f"AI analysis started: {os.path.basename(image_path)}")
    try:
        img, boxes = detect_boxes(image_path, min_area_ratio)
    except Exception as e:
        print(f"Error: {e}")
        return False
        
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    for count, box in enumerate(boxes):
        cropped = crop_and_warp(img, np.float32(box))
        out_filename = f"{name}_cropped_{count}{ext}" if count > 0 else f"{name}_cropped{ext}"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, cropped)
        print(f"Saved cropped image: {out_path}")
        
    if not boxes:
        print(f"AI found no images on scan: {image_path}")
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Automatic cropping of scanned photos using AI.")
    parser.add_argument("-i", "--input", default="input", help="Folder with photos to crop.")
    parser.add_argument("-o", "--output", default="output", help="Target folder for cropped photos.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    if not os.path.exists(args.input):
        print(f"Input folder '{args.input}' does not exist! Creating it.")
        os.makedirs(args.input)
        print("Place photos in the input folder and run the script again.")
        return
        
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    processed_count = 0
    
    for filename in os.listdir(args.input):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            path = os.path.join(args.input, filename)
            if process_image(path, args.output):
                processed_count += 1
                
    if processed_count == 0:
        print("No files to process.")
    else:
        print(f"Finished! Processed {processed_count} files.")

if __name__ == "__main__":
    main()
