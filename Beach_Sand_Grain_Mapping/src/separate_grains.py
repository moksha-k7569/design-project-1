
import cv2
import numpy as np
from pathlib import Path

# Main project folder
project_folder = Path(__file__).resolve().parent.parent

# Load scaled sand image
image_path = project_folder / "dataset" / "real_sand_scaled_01.jpeg"

image = cv2.imread(str(image_path))

if image is None:
    print("ERROR: Could not read the image.")

else:
    # Exclude ruler area
    sand = image[:, :1020].copy()

    # Convert to grayscale
    gray = cv2.cvtColor(sand, cv2.COLOR_BGR2GRAY)

    # Segment grains
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Make grains white
    grain_mask = cv2.bitwise_not(binary)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        grain_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distance transform
    distance = cv2.distanceTransform(
        opening, cv2.DIST_L2, 5
    )

    # Sure foreground
    _, sure_fg = cv2.threshold(
        distance,
        0.20 * distance.max(),
        255,
        cv2.THRESH_BINARY
    )

    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Create markers
    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(sand, markers)

    # Draw boundaries
    output = sand.copy()
    output[markers == -1] = [0, 255, 0]

    # Count detected regions
    grain_count = len(np.unique(markers[markers > 1]))

    print("Watershed separation completed!")
    print("Estimated separated regions:", grain_count)

    # Save result
    output_path = (
        project_folder / "results" /
        "real_sand_scaled_separated.png"
    )

    cv2.imwrite(str(output_path), output)

    print("Separated grain image saved successfully!")

    # Display result
    cv2.imshow("Separated Sand Grains", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
