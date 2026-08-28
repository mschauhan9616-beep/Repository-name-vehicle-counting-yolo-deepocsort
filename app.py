
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import subprocess
from collections import defaultdict, Counter


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Vehicle Counting System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Detection & Counting System")

st.write(
    "YOLO + Deep OC-SORT based vehicle detection, "
    "tracking and line-crossing counting."
)


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.60,
    value=0.25,
    step=0.05
)

image_size = st.sidebar.selectbox(
    "YOLO Image Size",
    [640, 960, 1280, 1536],
    index=2
)

line_position = st.sidebar.slider(
    "Counting Line Position",
    min_value=0.40,
    max_value=0.80,
    value=0.60,
    step=0.05
)


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):

    st.error(
        "Model not found at models/best.pt"
    )

    st.stop()


# =========================================================
# VIDEO UPLOAD
# =========================================================

uploaded_video = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded_video is not None:

    st.subheader("Original Video")

    st.video(uploaded_video)

    if st.button(
        "🚀 Start Vehicle Counting",
        type="primary"
    ):

        # =================================================
        # SAVE UPLOADED VIDEO
        # =================================================

        input_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_temp.write(
            uploaded_video.getvalue()
        )

        input_temp.close()

        input_path = input_temp.name


        # =================================================
        # FRESH MODEL
        # =================================================

        model = YOLO(MODEL_PATH)


        # =================================================
        # OPEN VIDEO
        # =================================================

        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():

            st.error(
                "Could not open uploaded video."
            )

            st.stop()


        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        st.write(
            f"Resolution: {width} × {height}"
        )

        st.write(
            f"Frames: {total_frames}"
        )


        # =================================================
        # ROAD ROI
        # =================================================

        road_roi = np.array([
            [
                (
                    int(width * 0.25),
                    int(height * 0.20)
                ),

                (
                    int(width * 0.78),
                    int(height * 0.20)
                ),

                (
                    int(width * 0.93),
                    int(height * 0.98)
                ),

                (
                    int(width * 0.05),
                    int(height * 0.98)
                )
            ]
        ], dtype=np.int32)


        frame_area = (
            width * height
        )


        # =================================================
        # COUNTING LINE
        # =================================================

        line_y = int(
            height * line_position
        )

        line_x1 = int(
            width * 0.15
        )

        line_x2 = int(
            width * 0.85
        )


        # =================================================
        # TEMP OUTPUT VIDEO
        # =================================================

        raw_output = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        raw_output_path = raw_output.name

        raw_output.close()


        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            raw_output_path,
            fourcc,
            fps,
            (width, height)
        )


        # =================================================
        # VARIABLES
        # =================================================

        previous_positions = {}

        counted_ids = set()

        track_class_votes = defaultdict(
            list
        )

        frame_number = 0


        # =================================================
        # STREAMLIT PROGRESS
        # =================================================

        progress_bar = st.progress(0)

        status_text = st.empty()

        live_total = st.empty()


        # =================================================
        # PROCESS VIDEO
        # =================================================

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1


            # =============================================
            # YOLO + DEEP OC-SORT
            # =============================================

            results = model.track(
                frame,
                persist=True,
                tracker="deepocsort.yaml",
                conf=confidence,
                imgsz=image_size,
                iou=0.50,
                verbose=False
            )


            result = results[0]

            annotated = frame.copy()


            # =============================================
            # TRACKS
            # =============================================

            if (
                result.boxes is not None
                and
                result.boxes.id is not None
            ):

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .tolist()
                )

                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                class_ids = (
                    result.boxes.cls
                    .int()
                    .cpu()
                    .tolist()
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .tolist()
                )


                for (
                    box,
                    track_id,
                    class_id,
                    conf
                ) in zip(
                    boxes,
                    track_ids,
                    class_ids,
                    confidences
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )


                    # =====================================
                    # BOTTOM-CENTER POINT
                    # =====================================

                    point_x = int(
                        (x1 + x2) / 2
                    )

                    point_y = int(
                        y2
                    )


                    # =====================================
                    # ROI FILTER
                    # =====================================

                    inside_road = (
                        cv2.pointPolygonTest(
                            road_roi[0],
                            (
                                point_x,
                                point_y
                            ),
                            False
                        )
                        >= 0
                    )


                    # =====================================
                    # LARGE FALSE BOX FILTER
                    # =====================================

                    box_area = (
                        (x2 - x1)
                        *
                        (y2 - y1)
                    )

                    area_ratio = (
                        box_area
                        /
                        frame_area
                    )

                    too_large = (
                        area_ratio > 0.12
                    )


                    if (
                        not inside_road
                        or
                        too_large
                    ):

                        continue


                    # =====================================
                    # CLASS VOTING
                    # =====================================

                    class_name = (
                        model.names[
                            class_id
                        ]
                    )

                    track_class_votes[
                        track_id
                    ].append(
                        class_name
                    )


                    # =====================================
                    # DRAW BOX
                    # =====================================

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )


                    label = (
                        f"ID:{track_id} "
                        f"{class_name} "
                        f"{conf:.2f}"
                    )


                    cv2.putText(
                        annotated,
                        label,
                        (
                            x1,
                            max(
                                y1 - 8,
                                20
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )


                    cv2.circle(
                        annotated,
                        (
                            point_x,
                            point_y
                        ),
                        5,
                        (0, 255, 255),
                        -1
                    )


                    # =====================================
                    # LINE CROSSING
                    # =====================================

                    if (
                        track_id
                        in previous_positions
                    ):

                        previous_x, previous_y = (
                            previous_positions[
                                track_id
                            ]
                        )


                        inside_line = (
                            line_x1
                            <= point_x
                            <= line_x2
                        )


                        # Selected road direction:
                        # count downward vehicles
                        crossed_down = (
                            previous_y
                            < line_y
                            and
                            point_y
                            >= line_y
                        )


                        if (
                            inside_line
                            and
                            crossed_down
                            and
                            track_id
                            not in counted_ids
                        ):

                            counted_ids.add(
                                track_id
                            )


                    previous_positions[
                        track_id
                    ] = (
                        point_x,
                        point_y
                    )


            # =============================================
            # DRAW ROI
            # =============================================

            cv2.polylines(
                annotated,
                road_roi,
                True,
                (0, 255, 255),
                2
            )


            # =============================================
            # COUNTING LINE
            # =============================================

            cv2.line(
                annotated,
                (
                    line_x1,
                    line_y
                ),
                (
                    line_x2,
                    line_y
                ),
                (0, 0, 255),
                4
            )


            cv2.putText(
                annotated,
                "COUNTING LINE",
                (
                    line_x1,
                    line_y - 12
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


            # =============================================
            # TOTAL
            # =============================================

            cv2.putText(
                annotated,
                f"TOTAL: {len(counted_ids)}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3
            )


            writer.write(
                annotated
            )


            # =============================================
            # STREAMLIT PROGRESS
            # =============================================

            if total_frames > 0:

                progress = min(
                    frame_number
                    /
                    total_frames,
                    1.0
                )

                progress_bar.progress(
                    progress
                )


            status_text.text(
                f"Processing frame "
                f"{frame_number}/"
                f"{total_frames}"
            )


            live_total.metric(
                "Vehicles Counted",
                len(counted_ids)
            )


        # =================================================
        # RELEASE
        # =================================================

        cap.release()

        writer.release()


        # =================================================
        # MAJORITY VOTING
        # =================================================

        final_class_counts = defaultdict(
            int
        )


        for track_id in counted_ids:

            votes = track_class_votes[
                track_id
            ]

            if len(votes) == 0:
                continue


            majority_class = (
                Counter(votes)
                .most_common(1)[0][0]
            )


            final_class_counts[
                majority_class
            ] += 1


        # =================================================
        # FINAL UI
        # =================================================

        st.success(
            "✅ Video processing completed!"
        )


        st.header(
            f"🚘 Total Vehicles: "
            f"{len(counted_ids)}"
        )


        # =================================================
        # CLASS TABLE
        # =================================================

        table_data = []


        for (
            class_name,
            count
        ) in final_class_counts.items():

            table_data.append(
                {
                    "Vehicle Type":
                        class_name,

                    "Count":
                        count
                }
            )


        if table_data:

            st.subheader(
                "Vehicle Count by Class"
            )

            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True
            )


        # =================================================
        # OUTPUT VIDEO
        # =================================================

        st.subheader(
            "Processed Video"
        )


        with open(
            raw_output_path,
            "rb"
        ) as video_file:

            video_bytes = (
                video_file.read()
            )


        st.download_button(
            label="⬇️ Download Processed Video",
            data=video_bytes,
            file_name="vehicle_counting_output.mp4",
            mime="video/mp4"
        )


        # =================================================
        # CLEAN INPUT
        # =================================================

        try:
            os.remove(input_path)

        except:
            pass
