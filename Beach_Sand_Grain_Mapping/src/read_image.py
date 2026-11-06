import cv2
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

image_path = project_folder / "dataset" / "real_sand_01.jpeg"

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Image not found!")
else:
    print("Real sand image loaded successfully!")
    print("Image width:", image.shape[1])
    print("Image height:", image.shape[0])

    cv2.imshow("Real Sand Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()