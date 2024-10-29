import cv2

WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)


def draw_line(image, p1, p2, color):
    # Kiểm tra kiểu dữ liệu của p1 và p2
    if isinstance(p1, tuple) and isinstance(p2, tuple) and len(p1) == 2 and len(p2) == 2:
        # Đảm bảo các phần tử là số nguyên
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        cv2.line(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)


def draw_keypoints(keypoints, frame):
    if len(keypoints) == 17:  # Đảm bảo đủ số điểm khớp
        # Lấy tọa độ của các điểm khớp

        draw_line(frame, keypoints[3], keypoints[4], WHITE_COLOR)
        draw_line(frame, keypoints[3], keypoints[5], WHITE_COLOR)
        draw_line(frame, keypoints[5], keypoints[7], WHITE_COLOR)
        draw_line(frame, keypoints[3], keypoints[9], WHITE_COLOR)
        draw_line(frame, keypoints[9], keypoints[11], WHITE_COLOR)
        draw_line(frame, keypoints[11], keypoints[13], WHITE_COLOR)

        draw_line(frame, keypoints[9], keypoints[10], WHITE_COLOR)

        draw_line(frame, keypoints[4], keypoints[6], WHITE_COLOR)
        draw_line(frame, keypoints[6], keypoints[8], WHITE_COLOR)
        draw_line(frame, keypoints[4], keypoints[10], WHITE_COLOR)
        draw_line(frame, keypoints[10], keypoints[12], WHITE_COLOR)
        draw_line(frame, keypoints[12], keypoints[14], WHITE_COLOR)

        # Vẽ các điểm khớp lên bức ảnh
        for i, (x, y) in enumerate(keypoints):
            if i in [0, 3, 4]:
                continue
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
