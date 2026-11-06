import cv2
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

image_path = project_folder / "dataset" / "real_sand_scaled_01.jpeg"

image = cv2.imread(str(image_path))

if image is None:
    print("Image not found!")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    output_path = project_folder / "results" / "real_sand_scaled_gray.png"

    cv2.imwrite(str(output_path), gray)

    print("Scaled sand grayscale image saved successfully!")
    print("Image dimensions:", gray.shape)

    cv2.imshow("Scaled Sand Grayscale", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()