
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Main project folder
project_folder = Path(__file__).resolve().parent.parent

# Read scaled sand image
image_path = project_folder / "dataset" / "real_sand_scaled_01.jpeg"
image = cv2.imread(str(image_path))

# Read calibration value
calibration_file = project_folder / "results" / "calibration.txt"

if image is None:
    print("ERROR: Could not read the image.")

elif not calibration_file.exists():
    print("ERROR: Calibration file not found.")

else:
    with open(calibration_file, "r") as file:
        lines = file.readlines()

    mm_per_pixel = float(lines[1].split(":")[1].strip())

    # Exclude ruler
    sand = image[:, :1020]

    # Convert to grayscale
    gray = cv2.cvtColor(sand, cv2.COLOR_BGR2GRAY)

    # Segment grains
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    grain_mask = cv2.bitwise_not(binary)

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        grain_mask, cv2.MORPH_OPEN, kernel, iterations=1
    )

    # Distance transform
    distance = cv2.distanceTransform(
        opening, cv2.DIST_L2, 5
    )

    # Separate grain regions
    _, sure_fg = cv2.threshold(
        distance, 0.05 * distance.max(), 255, 0
    )

    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)

    # Measure grains
    measurements = []

    for label in range(1, markers.max() + 1):
        grain = np.uint8(markers == label) * 255
        area_pixels = cv2.countNonZero(grain)

        if area_pixels < 20:
            continue

        diameter_pixels = np.sqrt(
            (4 * area_pixels) / np.pi
        )

        diameter_mm = diameter_pixels * mm_per_pixel
        area_mm2 = area_pixels * (mm_per_pixel ** 2)

        measurements.append({
            "Grain_ID": len(measurements) + 1,
            "Area_pixels": area_pixels,
            "Area_mm2": area_mm2,
            "Diameter_pixels": diameter_pixels,
            "Diameter_mm": diameter_mm
        })

    # Save CSV
    df = pd.DataFrame(measurements)

    results_folder = project_folder / "results"
    results_folder.mkdir(exist_ok=True)

    output_file = results_folder / "grain_measurements.csv"
    df.to_csv(output_file, index=False)

    print("Grain measurement completed!")
    print("Number of measured regions:", len(df))
    print("Calibration:", mm_per_pixel, "mm per pixel")
    print("Results saved to:", output_file)
