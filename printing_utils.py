import cv2
from settings import CONFIDENCE_LIMIT, LINE_THICKNESS
from animal_points import *


def add_connections(frame, points, width=1, height=1, artem_edition=False):
    for cur_line in all_points_connections:
        if not artem_edition:
            try :
                start_point = points[get_idx(cur_line.start_point)]
                end_point = points[get_idx(cur_line.end_point)]
            except Exception:
                continue
        else:
            start_point = points[get_idx_artem_edition(cur_line.start_point)]
            end_point = points[get_idx_artem_edition(cur_line.end_point)]

        if start_point.confidence > CONFIDENCE_LIMIT and end_point.confidence > CONFIDENCE_LIMIT:
            print(start_point, end_point)

            start_coord = int(start_point.x * width), int(start_point.y * height)
            end_coord = int(end_point.x * width), int(end_point.y * height)

            frame = cv2.line(frame, start_coord, end_coord, cur_line.connection_color, LINE_THICKNESS)
    return frame


def add_points(frame, points, width=1, height=1, m_color=(0, 255, 0), print_txt=False):
    for cur_point in points:
        if cur_point.confidence > CONFIDENCE_LIMIT:
            frame = cv2.circle(frame, (int(cur_point.x * width), int(cur_point.y * height)), radius=8,
                               color=m_color, thickness=-1)
            if print_txt:
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                # fontScale
                font_scale = 1
                # Blue color in BGR
                color = (255, 255, 0)
                # Line thickness of 2 px
                thickness = 1
                # Using cv2.putText() method
                frame = cv2.putText(frame, cur_point.name_of_point, (int(cur_point.x * width), int(cur_point.y * height)), font,
                                    font_scale, color, thickness, cv2.LINE_AA)
    return frame
