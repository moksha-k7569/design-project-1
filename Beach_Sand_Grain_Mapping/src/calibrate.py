
import cv2
import math
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

image_path = project_folder / "dataset" / "real_sand_scaled_01.jpeg"

image = cv2.imread(str(image_path))

if image is None:
    print("ERROR: Image not found!")
else:
    points = []

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print("Selected point:", x, y)

    cv2.namedWindow("Select Ruler Marks")
    cv2.setMouseCallback("Select Ruler Marks", select_points)

    print("Click two ruler marks exactly 1 cm apart.")
    print("Press ESC after selecting both points.")

    while True:
        display = image.copy()

        for point in points:
            cv2.circle(display, point, 7, (0, 0, 255), -1)

        cv2.imshow("Select Ruler Marks", display)

        key = cv2.waitKey(1) & 0xFF

        if len(points) == 2 or key == 27:
            break

    cv2.destroyAllWindows()

    if len(points) == 2:
        pixel_distance = math.dist(points[0], points[1])

        known_length_mm = 10.0

        mm_per_pixel = known_length_mm / pixel_distance

        print("Calibration completed!")
        print("Reference distance: 10 mm")
        print("Pixel distance:", pixel_distance)
        print("Millimetres per pixel:", mm_per_pixel)

        output_path = project_folder / "results" / "calibration.txt"

        with open(output_path, "w") as file:
            file.write(f"Pixels per 10 mm: {pixel_distance}\n")
            file.write(f"Millimetres per pixel: {mm_per_pixel}\n")

        print("Calibration saved successfully!")

    else:
        print("Calibration cancelled. Select two points.")
