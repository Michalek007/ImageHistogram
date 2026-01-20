import cv2
import os
import numpy as np

def save_image_as_c_array(image_path, output_file):
    # Load image using OpenCV
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError("Could not load image")

    height, width = image.shape[:2]

    with open(output_file, "w") as f:
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define IMG_WIDTH  {width}\n")
        f.write(f"#define IMG_HEIGHT {height}\n\n")

        # -------------------------
        # Grayscale Image
        # -------------------------
        if len(image.shape) == 2:
            f.write(f"const uint8_t image_gray[IMG_HEIGHT][IMG_WIDTH] = {{\n")

            for y in range(height):
                f.write("    { ")
                row = ", ".join(str(image[y, x]) for x in range(width))
                f.write(row)
                f.write(" },\n")

            f.write("};\n")

            # Compute histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.astype(np.uint32).flatten()  # Use uint32 to store counts safely
            f.write(f"const uint32_t hist_gray[256] = {{\n    " + ", ".join(map(str, hist)) + "\n};\n")

        # -------------------------
        # Color Image (BGR → RGB)
        # -------------------------
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            channels = {
                "R": image[:, :, 0],
                "G": image[:, :, 1],
                "B": image[:, :, 2],
            }

            for name, channel in channels.items():
                f.write(f"const uint8_t image_{name}[IMG_HEIGHT][IMG_WIDTH] = {{\n")

                for y in range(height):
                    f.write("    { ")
                    row = ", ".join(str(channel[y, x]) for x in range(width))
                    f.write(row)
                    f.write(" },\n")

                f.write("};\n\n")

    print(f"C array saved to: {os.path.abspath(output_file)}")


def save_image_histogram_as_c_array(image_path, output_file):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError("Could not load image")

    with open(output_file, "w") as f:
        f.write("#include <stdint.h>\n\n")
        # Compute histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.astype(np.uint32).flatten()  # Use uint32 to store counts safely
        f.write(f"const uint32_t hist_gray[256] = {{\n    " + ", ".join(map(str, hist)) + "\n};\n")


def save_image_as_raw(image_path, output_file):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Write raw bytes to file
    with open(output_file, "wb") as f:
        f.write(image.tobytes())


if __name__ == "__main__":
    # save_image_as_c_array("img.png", "image_data.c")
    save_image_as_raw("verybig.png",  "verybig")
    save_image_histogram_as_c_array("verybig.png",  "verybig.c")