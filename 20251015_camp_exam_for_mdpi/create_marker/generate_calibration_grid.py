import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

def create_calibration_grid():
    # A4 size in points (72 points per inch)
    width, height = A4
    
    # Create PDF
    c = canvas.Canvas("cal/cab_a4.pdf", pagesize=A4)
    
    # Grid parameters
    square_size = 30 * mm  # 30mm squares
    
    # Calculate number of squares that fit on A4
    cols = int(width // square_size)
    rows = int(height // square_size)
    
    # Center the grid
    start_x = (width - (cols * square_size)) / 2
    start_y = (height - (rows * square_size)) / 2
    
    # Draw checkerboard pattern
    for row in range(rows):
        for col in range(cols):
            x = start_x + col * square_size
            y = start_y + row * square_size
            
            # Alternate colors (black and white)
            if (row + col) % 2 == 0:
                c.setFillColorRGB(0, 0, 0)  # Black
                c.rect(x, y, square_size, square_size, fill=1)
    
    c.save()
    print(f"Created A4 calibration grid with {cols}x{rows} squares (20mm each) -> cal/cab_a4.pdf")

if __name__ == "__main__":
    create_calibration_grid()