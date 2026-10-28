import cv2 as cv


image_path = "sample_inputs/101_1.tif"

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if image is None:
    print("KHONG DOC DUOC ANH")
else:
    print("DOC ANH THANH CONG")
    print("Kich thuoc:", image.shape)
    print("Kieu du lieu:", image.dtype)

    cv.imwrite("output/test_101_1.png", image)
    print("Da luu anh test vao output/test_101_1.png")