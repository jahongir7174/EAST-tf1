# coding:utf-8
import csv
import os
from os import path as ops

import cv2
import numpy as np
import tensorflow as tf
from shapely.geometry import Polygon

from src.utils import config


def get_images(path):
    extensions = ['jpg', 'png']
    files = [('dataset/images/' + name) for name in os.listdir(path) if name[-3:] in extensions]
    return files


def load_annotation(p):
    _text_polygons = []
    _text_tags = []
    if not os.path.exists(p):
        return np.array(_text_polygons, dtype=np.float32)
    with open(p, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            _text_polygons.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                _text_tags.append(True)
            else:
                _text_tags.append(False)
        return np.array(_text_polygons, dtype=np.float32), np.array(_text_tags, dtype=np.bool)


def polygon_area(poly):
    edge = [(poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])]
    return np.sum(edge) / 2.


def check_and_validate_polygons(_polygons, _tags, _shape):
    (h, w) = _shape
    if _polygons.shape[0] == 0:
        return _polygons
    _polygons[:, :, 0] = np.clip(_polygons[:, :, 0], 0, w - 1)
    _polygons[:, :, 1] = np.clip(_polygons[:, :, 1], 0, h - 1)

    _validated_polygons = []
    _validated_tags = []
    for poly, tag in zip(_polygons, _tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        _validated_polygons.append(poly)
        _validated_tags.append(tag)
    return np.array(_validated_polygons), np.array(_validated_tags)


def crop_area(_image, _polygons, _tags, _crop_background=False, _max_tries=50):
    h, w, _ = _image.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in _polygons:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return _image, _polygons, _tags
    for i in range(_max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < config.min_crop_side_ratio * w or ymax - ymin < config.min_crop_side_ratio * h:
            continue
        if _polygons.shape[0] != 0:
            poly_axis_in_area = (_polygons[:, :, 0] >= xmin) & (_polygons[:, :, 0] <= xmax) \
                                & (_polygons[:, :, 1] >= ymin) & (_polygons[:, :, 1] <= ymax)
            selected_polygons = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polygons = []
        if len(selected_polygons) == 0:
            if _crop_background:
                return _image[ymin:ymax + 1, xmin:xmax + 1, :], _polygons[selected_polygons], _tags[selected_polygons]
            else:
                continue
        _image = _image[ymin:ymax + 1, xmin:xmax + 1, :]
        _polygons = _polygons[selected_polygons]
        _tags = _tags[selected_polygons]
        _polygons[:, :, 0] -= xmin
        _polygons[:, :, 1] -= ymin
        return _image, _polygons, _tags

    return _image, _polygons, _tags


def shrink_poly(poly, r):
    ratio = 0.3
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += ratio * r[0] * np.cos(theta)
        poly[0][1] += ratio * r[0] * np.sin(theta)
        poly[1][0] -= ratio * r[1] * np.cos(theta)
        poly[1][1] -= ratio * r[1] * np.sin(theta)
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += ratio * r[3] * np.cos(theta)
        poly[3][1] += ratio * r[3] * np.sin(theta)
        poly[2][0] -= ratio * r[2] * np.cos(theta)
        poly[2][1] -= ratio * r[2] * np.sin(theta)
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += ratio * r[0] * np.sin(theta)
        poly[0][1] += ratio * r[0] * np.cos(theta)
        poly[3][0] -= ratio * r[3] * np.sin(theta)
        poly[3][1] -= ratio * r[3] * np.cos(theta)
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += ratio * r[1] * np.sin(theta)
        poly[1][1] += ratio * r[1] * np.cos(theta)
        poly[2][0] -= ratio * r[2] * np.sin(theta)
        poly[2][1] -= ratio * r[2] * np.cos(theta)
    else:
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += ratio * r[0] * np.sin(theta)
        poly[0][1] += ratio * r[0] * np.cos(theta)
        poly[3][0] -= ratio * r[3] * np.sin(theta)
        poly[3][1] -= ratio * r[3] * np.cos(theta)
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += ratio * r[1] * np.sin(theta)
        poly[1][1] += ratio * r[1] * np.cos(theta)
        poly[2][0] -= ratio * r[2] * np.sin(theta)
        poly[2][1] -= ratio * r[2] * np.cos(theta)
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += ratio * r[0] * np.cos(theta)
        poly[0][1] += ratio * r[0] * np.sin(theta)
        poly[1][0] -= ratio * r[1] * np.cos(theta)
        poly[1][1] -= ratio * r[1] * np.sin(theta)
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += ratio * r[3] * np.cos(theta)
        poly[3][1] += ratio * r[3] * np.sin(theta)
        poly[2][0] -= ratio * r[2] * np.cos(theta)
        poly[2][1] -= ratio * r[2] * np.sin(theta)
    return poly


def point_dist_to_line(p1, p2, p3):
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fit_line(p1, p2):
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype=np.float32)


def line_vertical(line, point):
    if line[1] == 0:
        vertical = [0, -1, point[1]]
    else:
        if line[0] == 0:
            vertical = [1, 0, -point[0]]
        else:
            vertical = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
    return vertical


def rectangle_from_parallelogram(poly):
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_vertical = line_vertical(p2p3, p0)
            new_p3 = line_cross_point(p2p3, p2p3_vertical)
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_vertical = line_vertical(p0p1, p2)
            new_p1 = line_cross_point(p0p1, p0p1_vertical)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_vertical = line_vertical(p1p2, p0)
            new_p1 = line_cross_point(p1p2, p1p2_vertical)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_vertical = line_vertical(p0p3, p2)
            new_p3 = line_cross_point(p0p3, p0p3_vertical)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_vertical = line_vertical(p2p3, p1)
            new_p2 = line_cross_point(p2p3, p2p3_vertical)
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_vertical = line_vertical(p0p1, p3)
            new_p0 = line_cross_point(p0p1, p0p1_vertical)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_vertical = line_vertical(p0p3, p1)
            new_p0 = line_cross_point(p0p3, p0p3_vertical)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_vertical = line_vertical(p1p2, p3)
            new_p2 = line_cross_point(p1p2, p1p2_vertical)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        p_lowest_right = (p_lowest - 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def restore_rectangle_r_box(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2
        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2
        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))
        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2
        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin
        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2
        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2
        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))
        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2
        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin
        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_r_box(origin, geometry)


def generate_r_box(_shape, _polygons, _tags):
    h, w = _shape
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    training_mask = np.ones((h, w), dtype=np.uint8)
    for poly_idx, poly_tag in enumerate(zip(_polygons, _tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]
        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        shrunk_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrunk_poly, 1)
        cv2.fillPoly(poly_mask, shrunk_poly, poly_idx + 1)
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < config.min_text_size:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            new_p1 = p1
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            new_p0 = p0
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]
        rectangle = rectangle_from_parallelogram(parallelogram)
        rectangle, rotate_angle = sort_rectangle(rectangle)
        p0_rect, p1_rect, p2_rect, p3_rect = rectangle
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            geo_map[y, x, 4] = rotate_angle
    return score_map, geo_map, training_mask


def _bytes_feature(value):
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _is_valid_jpg_file(image_path):
    if not ops.exists(image_path):
        return False

    file = open(image_path, 'rb')
    data = file.read(11)
    if data[:4] != '\xff\xd8\xff\xe0' and data[:4] != '\xff\xd8\xff\xe1':
        file.close()
        return False
    if data[6:] != 'JFIF\0' and data[6:] != 'Exif\0':
        file.close()
        return False
    file.close()

    file = open(image_path, 'rb')
    file.seek(-2, 2)
    if file.read() != '\xff\xd9':
        file.close()
        return False

    file.close()

    return True


def __get_labels(_path):
    input_size = 512
    _image = cv2.imread(_path, 0)
    _image = _image[:, :, np.newaxis]
    h, w, _ = _image.shape
    txt_fn = config.training_label_path + 'gt_' + os.path.basename(_path).split('.')[0] + '.txt'
    text_polygons, text_tags = load_annotation(txt_fn)
    text_polygons, text_tags = check_and_validate_polygons(text_polygons, text_tags, (h, w))
    _image, text_polygons, text_tags = crop_area(_image, text_polygons, text_tags)

    h, w, _ = _image.shape
    new_h, new_w, _ = _image.shape
    max_h_w_i = np.max([new_h, new_w, input_size])
    im_padded = np.zeros((max_h_w_i, max_h_w_i, 1), dtype=np.uint8)
    im_padded[:new_h, :new_w, :] = _image.copy()
    _image = im_padded
    new_h, new_w, _ = _image.shape
    resize_h = input_size
    resize_w = input_size
    _image = cv2.resize(_image, dsize=(resize_w, resize_h))
    resize_ratio_3_x = resize_w / float(new_w)
    resize_ratio_3_y = resize_h / float(new_h)
    text_polygons[:, :, 0] *= resize_ratio_3_x
    text_polygons[:, :, 1] *= resize_ratio_3_y
    _image = _image[:, :, np.newaxis]
    new_h, new_w, _ = _image.shape
    _score_map, _geo_map, _training_mask = generate_r_box((new_h, new_w), text_polygons, text_tags)
    _score_map = _score_map[::4, ::4, np.newaxis].astype(np.float32)
    _geo_map = _geo_map[::4, ::4, :].astype(np.float32)
    _training_mask = _training_mask[::4, ::4, np.newaxis].astype(np.float32)
    return _image, _path, _score_map, _geo_map, _training_mask


def _write_tf_records(_queue, _sentinel):
    while True:
        sample_path = _queue.get()

        if sample_path == _sentinel:
            break
        if _is_valid_jpg_file(sample_path):
            continue

        try:
            print(sample_path)
            image, path, score_map, geo_map, training_mask = __get_labels(sample_path)
            image = image.tostring()
            score_map = score_map.tostring()
            geo_map = geo_map.tostring()
            training_mask = training_mask.tostring()

        except IOError as err:
            continue

        features = tf.train.Features(feature={'image': _bytes_feature(image),
                                              'path': _bytes_feature(path),
                                              'score_map': _bytes_feature(score_map),
                                              'geo_map': _bytes_feature(geo_map),
                                              'training_mask': _bytes_feature(training_mask)})
        tf_example = tf.train.Example(features=features)
        save_path = 'dataset/tf_records/' + str(os.path.basename(path).split('.')[0]) + '.tfrecords'
        writer = tf.io.TFRecordWriter(path=save_path)
        writer.write(tf_example.SerializeToString())
