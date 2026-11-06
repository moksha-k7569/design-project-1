import cv2
from pathlib import Path

# Main project folder
project_folder = Path(__file__).resolve().parent.parent

# Load scaled sand image
image_path = project_folder / "dataset" / "real_sand_scaled_01.jpeg"

image = cv2.imread(str(image_path))

if image is None:
    print("ERROR: Could not read the image.")

else:
    # Remove ruler area
    sand = image[:, :1020]

    # Convert to grayscale
    gray = cv2.cvtColor(sand, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Segment grains
    _, grain_mask = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grain_mask = cv2.morphologyEx(
        grain_mask, cv2.MORPH_OPEN, kernel
    )

    # Detect grain regions
    contours, _ = cv2.findContours(
        grain_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter small regions
    min_area = 20

    grain_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]

    print("Scaled sand grain detection completed!")
    print("Detected grain regions:", len(grain_contours))

    # Draw detected regions
    output = sand.copy()

    cv2.drawContours(
        output,
        grain_contours,
        -1,
        (0, 255, 0),
        1
    )

    # Save output
    output_path = project_folder / "results" / "real_sand_scaled_detected.png"

    cv2.imwrite(str(output_path), output)

    print("Detection image saved successfully!")

    cv2.imshow("Detected Sand Grain Regions", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()