import streamlit as st
import cv2 as cv
import numpy as np
import os

from finegerprint_pipline import fingerprint_pipline
from database import (
    create_table,
    add_person,
    get_all_persons
)
from matching import compare_minutiae


st.set_page_config(
    page_title="Fingerprint Recognition",
    page_icon="",
    layout="wide"
)


create_table()


st.title("Hệ thống nhận diện dấu vân tay")

st.write(
    "Hệ thống sử dụng OpenCV để xử lý ảnh vân tay, "
    "trích xuất minutiae và so sánh với dữ liệu đã đăng ký."
)


tab1, tab2, tab3 = st.tabs([
    "Đăng ký vân tay",
    "Xác định danh tính",
    "Database"
])


with tab1:

    st.header("Đăng ký dấu vân tay")

    col1, col2 = st.columns(2)

    with col1:

        person_code = st.text_input(
            "Mã người",
            placeholder="Ví dụ: SV004"
        )

    with col2:

        name = st.text_input(
            "Họ và tên",
            placeholder="Ví dụ: Nguyen Van D"
        )

    uploaded_file = st.file_uploader(
        "Chọn ảnh dấu vân tay",
        type=[
            "tif",
            "tiff",
            "bmp",
            "png",
            "jpg",
            "jpeg"
        ],
        key="register_upload"
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv.imdecode(
            file_bytes,
            cv.IMREAD_GRAYSCALE
        )

        if image is None:

            st.error(
                "Không thể đọc ảnh vân tay."
            )

        else:

            st.subheader("Ảnh đầu vào")

            st.image(
                image,
                caption="Ảnh vân tay",
                clamp=True
            )

            if st.button(
                "Đăng ký vân tay",
                type="primary"
            ):

                if person_code.strip() == "":

                    st.warning(
                        "Vui lòng nhập mã người."
                    )

                elif name.strip() == "":

                    st.warning(
                        "Vui lòng nhập họ tên."
                    )

                else:

                    persons = get_all_persons()

                    existed = False

                    for person in persons:

                        if (
                            person["person_code"]
                            == person_code.strip()
                        ):

                            existed = True
                            break

                    if existed:

                        st.error(
                            "Mã người này đã tồn tại!"
                        )

                    else:

                        with st.spinner(
                            "Đang xử lý ảnh vân tay..."
                        ):

                            try:

                                results, minutiae_points = (
                                    fingerprint_pipline(
                                        image
                                    )
                                )

                                if len(
                                    minutiae_points
                                ) == 0:

                                    st.error(
                                        "Không phát hiện được "
                                        "minutiae."
                                    )

                                else:

                                    add_person(
                                        person_code.strip(),
                                        name.strip(),
                                        minutiae_points
                                    )

                                    os.makedirs(
                                        "output",
                                        exist_ok=True
                                    )

                                    cv.imwrite(
                                        "output/"
                                        + "register_"
                                        + person_code.strip()
                                        + ".png",
                                        results
                                    )

                                    st.success(
                                        "Đăng ký vân tay "
                                        "thành công!"
                                    )

                                    st.info(
                                        f"Đã phát hiện "
                                        f"{len(minutiae_points)} "
                                        f"minutiae."
                                    )

                                    st.subheader(
                                        "Kết quả xử lý ảnh"
                                    )

                                    st.image(
                                        results,
                                        caption=(
                                            "Pipeline xử lý "
                                            "vân tay"
                                        ),
                                        clamp=True
                                    )

                            except Exception as e:

                                st.error(
                                    "Có lỗi khi xử lý: "
                                    + str(e)
                                )


with tab2:

    st.header("Xác định danh tính")

    uploaded_file_match = st.file_uploader(
        "Chọn ảnh vân tay cần nhận diện",
        type=[
            "tif",
            "tiff",
            "bmp",
            "png",
            "jpg",
            "jpeg"
        ],
        key="matching_upload"
    )

    if uploaded_file_match is not None:

        file_bytes = np.asarray(
            bytearray(
                uploaded_file_match.read()
            ),
            dtype=np.uint8
        )

        image = cv.imdecode(
            file_bytes,
            cv.IMREAD_GRAYSCALE
        )

        if image is None:

            st.error(
                "Không thể đọc ảnh."
            )

        else:

            st.subheader(
                "Ảnh vân tay cần nhận diện"
            )

            st.image(
                image,
                caption="Ảnh đầu vào",
                clamp=True
            )

            if st.button(
                "🔍 Nhận diện",
                type="primary"
            ):

                persons = get_all_persons()

                if len(persons) == 0:

                    st.warning(
                        "Database chưa có người nào."
                    )

                else:

                    with st.spinner(
                        "Đang phân tích và nhận diện..."
                    ):

                        try:

                            results, input_points = (
                                fingerprint_pipline(
                                    image
                                )
                            )

                            st.info(
                                f"Ảnh đầu vào có "
                                f"{len(input_points)} "
                                f"minutiae."
                            )

                            best_person = None
                            best_score = 0.0

                            for person in persons:

                                score = compare_minutiae(
                                    input_points,
                                    person["template"]
                                )

                                if score > best_score:

                                    best_score = score
                                    best_person = person

                            st.subheader(
                                "Kết quả nhận diện"
                            )

                            threshold = 30

                            if best_person is not None:

                                if best_score >= threshold:

                                    col1, col2, col3 = (
                                        st.columns(3)
                                    )

                                    with col1:

                                        st.metric(
                                            "Mã người",
                                            best_person[
                                                "person_code"
                                            ]
                                        )

                                    with col2:

                                        st.metric(
                                            "Họ tên",
                                            best_person[
                                                "name"
                                            ]
                                        )

                                    with col3:

                                        st.metric(
                                            "Similarity",
                                            f"{best_score:.2f}%"
                                        )

                                    st.success(
                                        "MATCH - "
                                        "Đã xác định được "
                                        "danh tính."
                                    )

                                else:

                                    st.warning(
                                        "UNKNOWN - "
                                        "Không tìm thấy "
                                        "danh tính phù hợp."
                                    )

                                    st.metric(
                                        "Độ tương đồng cao nhất",
                                        f"{best_score:.2f}%"
                                    )

                                    st.caption(
                                        f"Mẫu gần nhất: "
                                        f"{best_person['person_code']} - "
                                        f"{best_person['name']}"
                                    )

                            st.subheader(
                                "Kết quả xử lý ảnh"
                            )

                            st.image(
                                results,
                                caption=(
                                    "Pipeline xử lý "
                                    "ảnh vân tay"
                                ),
                                clamp=True
                            )

                            os.makedirs(
                                "output",
                                exist_ok=True
                            )

                            cv.imwrite(
                                "output/"
                                "matching_result.png",
                                results
                            )

                        except Exception as e:

                            st.error(
                                "Có lỗi khi nhận diện: "
                                + str(e)
                            )


with tab3:

    st.header("🗄️ Danh sách dấu vân tay")

    persons = get_all_persons()

    if len(persons) == 0:

        st.info(
            "Database hiện chưa có dữ liệu."
        )

    else:

        st.write(
            f"Hiện có **{len(persons)}** "
            f"người đã đăng ký."
        )

        for person in persons:

            with st.container():

                col1, col2, col3, col4 = (
                    st.columns([1, 2, 2, 1])
                )

                with col1:

                    st.write(
                        f"**ID:** "
                        f"{person['id']}"
                    )

                with col2:

                    st.write(
                        f"**Mã:** "
                        f"{person['person_code']}"
                    )

                with col3:

                    st.write(
                        f"**Tên:** "
                        f"{person['name']}"
                    )

                with col4:

                    st.write(
                        f"**Minutiae:** "
                        f"{len(person['template'])}"
                    )

                st.divider()

