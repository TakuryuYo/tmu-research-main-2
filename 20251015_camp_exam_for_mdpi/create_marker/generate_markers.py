import cv2
import numpy as np
import os

def generate_aruco_markers():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    for marker_id in range(50, 101):
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, 200)
        filename = f"markers/02{marker_id:d}.png"
        cv2.imwrite(filename, marker_image)
        print(f"Generated marker {marker_id} -> {filename}")

if __name__ == "__main__":
    generate_aruco_markers()