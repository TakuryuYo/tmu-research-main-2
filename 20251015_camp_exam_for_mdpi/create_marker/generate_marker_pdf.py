import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Image
import io
from PIL import Image as PILImage

def create_marker_pdf(start_id, marker_size_mm=20):
    # A4 size in points (72 points per inch)
    width, height = A4
    
    # Margin
    margin = 5 * mm
    
    # Marker IDs for corners (clockwise from top-right)
    marker_ids = [start_id, start_id+1, start_id+2, start_id+3]
    
    # Create PDF with filename based on marker IDs
    filename = f"marker_pdf/markers_a4_{marker_ids[0]}_{marker_ids[-1]}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    
    # Marker size in points
    marker_size = marker_size_mm * mm
    
    # Generate ArUco markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Corner positions with margin (clockwise from top-right)
    positions = [
        (width - marker_size - margin, height - marker_size - margin),  # top-right
        (width - marker_size - margin, margin),                        # bottom-right
        (margin, margin),                                               # bottom-left
        (margin, height - marker_size - margin)                        # top-left
    ]
    
    # Generate and place markers
    for i, (marker_id, (x, y)) in enumerate(zip(marker_ids, positions)):
        # Generate marker image
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, 200)
        
        # Convert OpenCV image to PIL Image
        pil_image = PILImage.fromarray(marker_image)
        
        # Save to temporary file
        temp_filename = f"temp_marker_{marker_id}.png"
        pil_image.save(temp_filename)
        
        # Draw marker on PDF
        c.drawImage(temp_filename, x, y, width=marker_size, height=marker_size)
        
        # Add marker ID text below marker
        text_x = x + marker_size / 2
        text_y = y - 10
        c.drawCentredString(text_x, text_y, f"ID: {marker_id}")
    
    c.save()
    
    # Clean up temporary files
    import os
    for marker_id in marker_ids:
        temp_filename = f"temp_marker_{marker_id}.png"
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    print(f"Created A4 PDF with ArUco markers at corners ({marker_size_mm}mm each) -> {filename}")
    print("Marker placement (clockwise from top-right):")
    for i, marker_id in enumerate(marker_ids):
        corner_names = ["top-right", "bottom-right", "bottom-left", "top-left"]
        print(f"  {corner_names[i]}: ID {marker_id}")
    print()

def generate_all_marker_pdfs():
    # Generate PDFs for marker IDs 50-100
    for start_id in range(50, 101, 4):
        # Check if we have 4 markers available
        if start_id + 3 <= 100:
            create_marker_pdf(start_id, marker_size_mm=20)
        else:
            print(f"Skipping markers starting from {start_id} (not enough for 4 corners)")

if __name__ == "__main__":
    generate_all_marker_pdfs()