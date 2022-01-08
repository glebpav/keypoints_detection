from settings import *


class Animals_points:
    def __init__(self, x, y, confidence, name_of_point):
        self.x = x
        self.y = y
        self.confidence = confidence
        self.name_of_point = name_of_point

    def __str__(self):
        return f"[confidence = {self.confidence}, x = {self.x}, y = {self.y}, name of point = {self.name_of_point}]"


class Points_connection:
    def __init__(self, start_point, end_point, connection_color):
        self.start_point = start_point
        self.end_point = end_point
        self.connection_color = connection_color


animals_points_classes_artem_edition = [
    'front_right_knee', 'throat_base', 'neck_end', 'tail_base', 'nose', 'left_eye', 'front_left_thai',
    'front_left_knee', 'back_left_paw', 'left_earend', 'lower_jaw', 'back_right_knee', 'right_eye', 'left_earbase',
    'front_left_paw', 'right_earbase', 'back_left_thai', 'back_middle', 'tail_end', 'upper_jaw', 'back_right_thai',
    'right_earend', 'mouth_end_left', 'mouth_end_right', 'back_left_knee', 'front_right_paw', 'neck_base', 'throat_end',
    'front_right_thai', 'back_right_paw', ]

animals_points_classes = [
    'throat_base', 'back_right_knee', 'tail_end', 'front_left_thai', 'lower_jaw', 'back_left_knee', 'back_right_paw',
    'back_left_paw', 'back_middle', 'mouth_end_right', 'left_earbase', 'front_left_paw', 'right_earbase',
    'front_left_knee', 'throat_end', 'back_right_thai', 'neck_end', 'upper_jaw', 'neck_base', 'tail_base', 'nose',
    'mouth_end_left', 'left_earend', 'front_right_knee', 'front_right_thai', 'left_eye', 'right_eye', 'right_earend',
    'front_right_paw', 'back_left_thai',
]

all_points_connections = {
    # back legs
    Points_connection('back_left_paw', 'back_left_knee', LEGS_COLOR),
    Points_connection('back_right_paw', 'back_right_knee', LEGS_COLOR),
    Points_connection('back_left_thai', 'back_left_knee', LEGS_COLOR),
    Points_connection('back_right_thai', 'back_right_knee', LEGS_COLOR),

    # front legs
    Points_connection('front_left_paw', 'front_left_knee', LEGS_COLOR),
    Points_connection('front_right_paw', 'front_right_knee', LEGS_COLOR),
    Points_connection('front_left_thai', 'front_left_knee', LEGS_COLOR),
    Points_connection('front_right_thai', 'front_right_knee', LEGS_COLOR),

    # roжа
    Points_connection('mouth_end_right', 'upper_jaw', HEAD_COLOR),
    Points_connection('mouth_end_left', 'upper_jaw', HEAD_COLOR),
    Points_connection('lower_jaw', 'upper_jaw', HEAD_COLOR),
    Points_connection('nose', 'upper_jaw', HEAD_COLOR),
    Points_connection('nose', 'right_eye', HEAD_COLOR),
    Points_connection('nose', 'left_eye', HEAD_COLOR),
    Points_connection('right_eye', 'left_eye', HEAD_COLOR),

    # ears
    Points_connection('right_earbase', 'right_eye', HEAD_COLOR),
    Points_connection('left_earbase', 'left_eye', HEAD_COLOR),
    Points_connection('right_earbase', 'right_earend', HEAD_COLOR),
    Points_connection('left_earbase', 'left_earend', HEAD_COLOR),

    # antlers
    Points_connection('left_earbase', 'left_antler_base', HEAD_COLOR),
    Points_connection('left_antler_base', 'left_antler_end', HEAD_COLOR),
    Points_connection('right_earbase', 'right_antler_base', HEAD_COLOR),
    Points_connection('right_antler_base', 'right_antler_end', HEAD_COLOR),

    # body
    Points_connection('neck_base', 'neck_end', BODY_COLOR),
    Points_connection('throat_base', 'throat_end', BODY_COLOR),
    Points_connection('back_base', 'throat_end', BODY_COLOR),

    Points_connection('back_base', 'back_middle', BODY_COLOR),
    Points_connection('back_end', 'back_middle', BODY_COLOR),
    Points_connection('back_end', 'body_middle_left', BODY_COLOR),
    Points_connection('back_end', 'body_middle_right', BODY_COLOR),
    Points_connection('back_base', 'body_middle_right', BODY_COLOR),
    Points_connection('back_base', 'body_middle_left', BODY_COLOR),

    Points_connection('back_end', 'tail_base', BODY_COLOR),
    Points_connection('tail_end', 'tail_base', BODY_COLOR),

    Points_connection('body_middle_right', 'front_right_thai', BODY_COLOR),
    Points_connection('body_middle_right', 'back_right_thai', BODY_COLOR),
    Points_connection('body_middle_left', 'front_left_thai', BODY_COLOR),
    Points_connection('body_middle_left', 'back_left_thai', BODY_COLOR),

    Points_connection('belly_bottom', 'front_right_thai', BODY_COLOR),
    Points_connection('belly_bottom', 'back_right_thai', BODY_COLOR),
    Points_connection('belly_bottom', 'front_left_thai', BODY_COLOR),
    Points_connection('belly_bottom', 'back_left_thai', BODY_COLOR), }


def get_idx(point):
    return animals_points_classes.index(point)


def get_idx_artem_edition(point):
    return animals_points_classes_artem_edition.index(point)
