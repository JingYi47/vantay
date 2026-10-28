import cv2 as cv

from finegerprint_pipline import fingerprint_pipline
from database import create_table, add_person


def register_fingerprint(image_path, person_code, name):

    # Đọc ảnh vân tay
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if image is None:
        print("Khong the doc anh:", image_path)
        return False

    print("Dang xu ly anh van tay...")

    # Chạy pipeline xử lý ảnh
    results, minutiae_points = fingerprint_pipline(image)

    print(
        f"Da phat hien {len(minutiae_points)} minutiae"
    )

    # Kiểm tra có phát hiện minutiae hay không
    if len(minutiae_points) == 0:
        print("Khong tim thay minutiae!")
        return False

    # Lưu vào database
    add_person(
        person_code,
        name,
        minutiae_points
    )

    print("Dang ky van tay thanh cong!")
    print("Ma nguoi:", person_code)
    print("Ho ten:", name)

    # Lưu ảnh pipeline để xem kết quả
    cv.imwrite(
        f"output/register_{person_code}.png",
        results
    )

    return True


if __name__ == "__main__":

    # Tạo database/table nếu chưa có
    create_table()

   
    # THAY 3 THÔNG TIN NÀY
    

    image_path = "sample_inputs/103_1.tif"

    person_code = "SV003"

    name = "Le Van C"

   

    register_fingerprint(
        image_path,
        person_code,
        name
    )