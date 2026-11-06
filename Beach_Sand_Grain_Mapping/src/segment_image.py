import cv2
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

image_path = project_folder / "dataset" / "real_sand_scaled_01.jpeg"

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Image not found!")
else:
    # Remove ruler from processing area
    sand = image[:, :1020]

    # Convert to grayscale
    gray = cv2.cvtColor(sand, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding
    _, segmented = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    output_path = project_folder / "results" / "real_sand_scaled_segmented.png"

    cv2.imwrite(str(output_path), segmented)

    print("Sand-only segmentation completed!")
    print("Ruler excluded from processing.")

    cv2.imshow("Sand Only Segmentation", segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()