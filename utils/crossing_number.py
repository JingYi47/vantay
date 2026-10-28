import cv2 as cv
import numpy as np


def minutiae_at(pixels, i, j, kernel_size):
    """
    Detect ridge endings and ridge bifurcations
    using the Crossing Number method.
    """

    if pixels[i][j] == 1:

        if kernel_size == 3:
            cells = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, 1),
                (1, 1), (1, 0), (1, -1),
                (0, -1),
                (-1, -1)
            ]
        else:
            cells = [
                (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                (-1, 2), (0, 2), (1, 2),
                (2, 2), (2, 1), (2, 0), (2, -1), (2, -2),
                (1, -2), (0, -2), (-1, -2),
                (-2, -2)
            ]

        values = [pixels[i + l][j + k] for k, l in cells]

        crossings = 0

        for k in range(len(values) - 1):
            crossings += abs(values[k] - values[k + 1])

        crossings //= 2

        if crossings == 1:
            return "ending"

        if crossings == 3:
            return "bifurcation"

    return "none"


def calculate_minutiaes(im, kernel_size=3, return_points=False):
    """
    Detect minutiae from a skeletonized fingerprint.

    Parameters
    ----------
    im : numpy.ndarray
        Skeletonized fingerprint image.

    kernel_size : int
        Neighborhood size.

    return_points : bool
        False -> return visualization image.
        True  -> return visualization image + minutiae data.

    Returns
    -------
    result : numpy.ndarray
        Fingerprint image with detected minutiae.

    minutiae_points : list
        List containing:
        x, y and type of each minutia.
    """

    # Convert skeleton image to binary
    binary_image = np.zeros_like(im)

    binary_image[im < 10] = 1

    binary_image = binary_image.astype(np.int8)

    height, width = im.shape

    # Image used for visualization
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

    colors = {
        "ending": (150, 0, 0),
        "bifurcation": (0, 150, 0)
    }

    # Store minutiae information
    minutiae_points = []

    # Iterate through fingerprint pixels
    for y in range(1, height - 1):

        for x in range(1, width - 1):

            minutiae = minutiae_at(
                binary_image,
                y,
                x,
                kernel_size
            )

            if minutiae != "none":

                # Draw minutiae on image
                cv.circle(
                    result,
                    (x, y),
                    radius=2,
                    color=colors[minutiae],
                    thickness=2
                )

                # Save minutiae information
                minutiae_points.append({
                    "x": int(x),
                    "y": int(y),
                    "type": minutiae
                })

    # Remove duplicated / very close minutiae
    filtered_points = []

    min_distance = 8

    for point in minutiae_points:

        duplicate = False

        for existing in filtered_points:

            distance = np.sqrt(
                (point["x"] - existing["x"]) ** 2 +
                (point["y"] - existing["y"]) ** 2
            )

            if distance < min_distance:
                duplicate = True
                break

        if not duplicate:
            filtered_points.append(point)

    if return_points:
        return result, filtered_points

    return result