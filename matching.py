import cv2 as cv
import math

from finegerprint_pipline import fingerprint_pipline
from database import create_table, get_all_persons


def distance(point1, point2):
    """
    Tính khoảng cách Euclid giữa 2 minutiae.
    """

    return math.sqrt(
        (point1["x"] - point2["x"]) ** 2
        +
        (point1["y"] - point2["y"]) ** 2
    )


def compare_minutiae(input_points, database_points):
    """
    So sánh minutiae của ảnh đầu vào
    với minutiae trong database.

    Cải tiến:
    - Chuẩn hóa tọa độ theo tâm của tập minutiae.
    - Cho phép sai lệch vị trí giữa hai lần chụp.
    - Chỉ ghép các minutiae cùng loại:
      ending với ending
      bifurcation với bifurcation
    """

    if len(input_points) == 0 or len(database_points) == 0:
        return 0.0

    # =====================================================
    # 1. TÍNH TÂM CỦA CÁC MINUTIAE ẢNH ĐẦU VÀO
    # =====================================================

    input_center_x = sum(
        point["x"] for point in input_points
    ) / len(input_points)

    input_center_y = sum(
        point["y"] for point in input_points
    ) / len(input_points)

    # =====================================================
    # 2. TÍNH TÂM CỦA CÁC MINUTIAE TRONG DATABASE
    # =====================================================

    database_center_x = sum(
        point["x"] for point in database_points
    ) / len(database_points)

    database_center_y = sum(
        point["y"] for point in database_points
    ) / len(database_points)

    # =====================================================
    # 3. CHUẨN HÓA TỌA ĐỘ
    #
    # Thay vì dùng tọa độ x,y ban đầu,
    # ta lấy khoảng cách của mỗi điểm so với tâm.
    #
    # Điều này giúp giảm ảnh hưởng khi ảnh bị dịch chuyển.
    # =====================================================

    normalized_input = []

    for point in input_points:

        normalized_input.append({
            "x": point["x"] - input_center_x,
            "y": point["y"] - input_center_y,
            "type": point["type"]
        })

    normalized_database = []

    for point in database_points:

        normalized_database.append({
            "x": point["x"] - database_center_x,
            "y": point["y"] - database_center_y,
            "type": point["type"]
        })

    # =====================================================
    # 4. MATCHING MINUTIAE
    # =====================================================

    matched = 0

    # Sai lệch vị trí cho phép
    max_distance = 25

    # Lưu các minutiae database đã được sử dụng
    used_database_points = set()

    # Duyệt từng minutiae của ảnh đầu vào
    for input_point in normalized_input:

        best_index = None
        best_distance = float("inf")

        # Tìm minutiae gần nhất trong database
        for i, database_point in enumerate(
            normalized_database
        ):

            # Không sử dụng lại cùng một minutia
            if i in used_database_points:
                continue

            # Hai minutiae phải cùng loại
            if input_point["type"] != database_point["type"]:
                continue

            # Tính dx
            dx = (
                input_point["x"]
                -
                database_point["x"]
            )

            # Tính dy
            dy = (
                input_point["y"]
                -
                database_point["y"]
            )

            # Khoảng cách Euclid
            current_distance = math.sqrt(
                dx ** 2 + dy ** 2
            )

            # Lấy điểm gần nhất
            if current_distance < best_distance:

                best_distance = current_distance
                best_index = i

        # Nếu tìm được điểm phù hợp
        if (
            best_index is not None
            and best_distance <= max_distance
        ):

            matched += 1

            used_database_points.add(
                best_index
            )

   
    # 5. TÍNH ĐỘ TƯƠNG ĐỒNG
   

    max_points = max(
        len(input_points),
        len(database_points)
    )

    similarity = (
        matched / max_points
    ) * 100

    return similarity


def identify_fingerprint(image_path):
    """
    Xác định danh tính của ảnh vân tay.

    Ảnh đầu vào sẽ được:
    1. Đọc bằng OpenCV.
    2. Chạy qua fingerprint pipeline.
    3. Trích xuất minutiae.
    4. So sánh với tất cả người trong database.
    5. Chọn người có similarity cao nhất.
    """

    print("\nDang xu ly anh van tay...")

    
    # 1. ĐỌC ẢNH
   

    image = cv.imread(
        image_path,
        cv.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            "Khong the doc anh:",
            image_path
        )

        return

    
    # 2. CHẠY PIPELINE XỬ LÝ ẢNH
   

    results, input_points = fingerprint_pipline(
        image
    )

    print(
        f"Anh dau vao co "
        f"{len(input_points)} minutiae"
    )

    
    # 3. LẤY DỮ LIỆU TỪ DATABASE
    

    persons = get_all_persons()

    if len(persons) == 0:

        print(
            "Database dang trong!"
        )

        return

   
    # 4. TÌM NGƯỜI PHÙ HỢP NHẤT
    

    best_person = None
    best_score = 0.0

    print("\n===== SO SANH =====")

    # So sánh với TẤT CẢ người
    for person in persons:

        score = compare_minutiae(
            input_points,
            person["template"]
        )

        print(
            f'{person["person_code"]} - '
            f'{person["name"]}: '
            f'{score:.2f}%'
        )

        # Nếu score cao hơn kết quả hiện tại
        if score > best_score:

            best_score = score
            best_person = person

    
    # 5. HIỂN THỊ KẾT QUẢ
    

    print(
        "\n===== KET QUA NHAN DIEN ====="
    )

    if best_person is None:

        print(
            "Khong tim thay nguoi phu hop."
        )

    else:

        print(
            "Ma:",
            best_person["person_code"]
        )

        print(
            "Ten:",
            best_person["name"]
        )

        print(
            f"Similarity: "
            f"{best_score:.2f}%"
        )

        
        # NGƯỠNG NHẬN DIỆN
       

        threshold = 30

        if best_score >= threshold:

            print(
                "Trang thai: MATCH"
            )

        else:

            print(
                "Trang thai: UNKNOWN"
            )

    
    # 6. LƯU ẢNH KẾT QUẢ PIPELINE
   

    cv.imwrite(
        "output/matching_result.png",
        results
    )

    print(
        "\nDa luu ket qua:"
        " output/matching_result.png"
    )


if __name__ == "__main__":

   
    # TẠO DATABASE NẾU CHƯA CÓ
   

    create_table()

    
    # ẢNH CẦN NHẬN DIỆN
    #
    # Hiện tại đang test 102_2.tif
    #
    # 102_1.tif đã được đăng ký cho SV002
    # nên 102_2.tif phải được nhận diện là SV002
    # =====================================================

    image_path = "sample_inputs/102_3.tif"

    # =====================================================

    identify_fingerprint(
        image_path
    )