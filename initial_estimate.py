import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

from image_utils import *
from typing import Tuple
import os

DETECTOR = cv2.AKAZE_create
NORM = cv2.NORM_HAMMING
DEVIATION_THRESHOLD = 0.5
PNP_THRESHOLD = 1
RANSAC_SUCCESS_PROB = 0.9999
RANSAC_NUM_SAMPLES = 4

DATA_PATH = os.path.join("VAN_ex", "dataset", "sequences", "00")
POSES_PATH = os.path.join("VAN_ex", "dataset", "poses")


def read_images(idx: int, color: ImageColor) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """
    This function read the image numbered idx.
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(os.path.join(DATA_PATH, 'image_0', img_name), color)
    img2 = cv2.imread(os.path.join(DATA_PATH, 'image_1', img_name), color)
    return img1, img2


def show_key_points(idx: int, kps1: NDArray[KeyPoint], kps2: NDArray[KeyPoint]) -> None:
    """
    This function present given key points on the left and right images.
    """
    img1, img2 = read_images(idx, ImageColor.RGB)
    cv2.drawKeypoints(img1, kps1[500:], img1, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imshow("Output Image 1", img1)
    cv2.drawKeypoints(img2, kps2[500:], img2, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imshow("Output Image 2", img2)
    cv2.waitKey(0)


def detect_key_points(idx: int) -> Tuple[Image, Image]:
    """
    This function finds key points on the left and right images numbered idx.
    """
    img1_mat, img2_mat = read_images(idx, ImageColor.GRAY)
    detector = DETECTOR()
    kps1, des1 = detector.detectAndCompute(img1_mat, None)
    kps2, des2 = detector.detectAndCompute(img2_mat, None)
    img1 = Image(img1_mat, np.array(kps1), des1)
    img2 = Image(img2_mat, np.array(kps2), des2)
    # show_key_points(idx, img1.kps, img2.kps)
    return img1, img2


def match_key_points(img1: Image, img2: Image, set_matches_idx: bool = True) -> Tuple[NDArray[DMatch], List[int], List[int]]:
    """
    This function finds matches between img1 and img2, and set the relevant key points indices of img1 and img2.
    """
    brute_force = cv2.BFMatcher(NORM, crossCheck=True)
    matches = np.array(brute_force.match(img1.desc, img2.desc))
    img1_kps_to_matches_idx = None
    img2_kps_to_matches_idx = None
    if set_matches_idx:
        img1_kps_to_matches_idx = np.full((len(img1.kps),), -1)
        img2_kps_to_matches_idx = np.full((len(img2.kps),), -1)
        img1_matches_idx, img2_matches_idx = [], []
        for i, match in enumerate(matches):
            img1_kps_to_matches_idx[match.queryIdx] = i
            img2_kps_to_matches_idx[match.trainIdx] = i
            img1_matches_idx.append(match.queryIdx)
            img2_matches_idx.append(match.trainIdx)
        img1.set_matches_kps_idx(img1_matches_idx)
        img2.set_matches_kps_idx(img2_matches_idx)
    # show_matches(img1, img2, matches)
    return matches, img1_kps_to_matches_idx, img2_kps_to_matches_idx


def show_matches(img1: Image, img2: Image, matches: NDArray[DMatch]) -> None:
    """
    This function presents the given matches between img1 and img2.
    """
    random_matches = matches[np.random.randint(len(matches), size=20)]
    res = np.empty((max(img1.mat.shape[0], img2.mat.shape[0]), img1.mat.shape[1] + img2.mat.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1.mat, img1.kps, img2.mat, img2.kps, random_matches, res, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Output matches", res)  # 1.3
    cv2.waitKey(0)


def print_feature_descriptors(descriptors1: NDArray[np.uint8], descriptors2: NDArray[np.uint8]) -> None:
    """
    This function print the given descriptors arrays.
    """
    print("The first two feature descriptors of image 1:")
    print(descriptors1[:2])
    print("The first two feature descriptors of image 2:")
    print(descriptors2[:2])


def significance_test(img1: Image, img2: Image) -> None:
    """
    This function is for rejecting matches that have key points which can match to many key points (instead of to one unique key point) by a ratio.
    :param img1:
    :param img2:
    :return:
    """
    res = np.empty((max(img1.mat.shape[0], img2.mat.shape[0]), img1.mat.shape[1] + img2.mat.shape[1], 3), dtype=np.uint8)
    brute_force = cv2.BFMatcher(cv2.NORM_L1)
    matches = brute_force.knnMatch(img1.desc, img2.desc, k=2)
    ratio = 0.5
    good_matches = np.array([m1 for m1, m2 in matches if m1.distance < ratio * m2.distance])
    random_matches = good_matches[np.random.randint(len(good_matches), size=20)]
    cv2.drawMatches(img1.mat, img1.kps, img2.mat, img2.kps, random_matches, res, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Output random good matches", res)  # 1.4
    cv2.waitKey(0)


def plot_histogram(deviations: FloatNDArray) -> None:
    """
    This function plots the histogram of the deviations between the y pixels of two key points from that matched in a stereo image.
    :param deviations:
    :return:
    """
    plt.hist(deviations, 50)
    plt.title("Histogram of deviations between matches")
    plt.ylabel("Number of matches")
    plt.xlabel("Deviation from rectified stereo pattern")
    plt.show()


def histogram_pattern(stereo_pair: StereoPair) -> FloatNDArray:
    """
    This function computes the deviations between the y pixels of two key points from that matched in a stereo image.
    :param stereo_pair:
    :return:
    """
    deviations = np.zeros(len(stereo_pair.matches))
    for i, match in enumerate(stereo_pair.matches):
        y1 = stereo_pair.left_image.kps[match.queryIdx].pt[1]
        y2 = stereo_pair.right_image.kps[match.trainIdx].pt[1]
        deviations[i] = abs(y2 - y1)
    # print(f"The percentage of matches that devaite by more than {DEVIATION_THRESHOLD} pixel:",
    #       round(100 * (len(deviations[deviations > DEVIATION_THRESHOLD]) / len(
    #           deviations)), 2))
    # plot_histogram(deviations)
    return deviations


def pattern_reject_matches(deviations: FloatNDArray, stereo_pair: StereoPair) -> None:
    """
    This function rejects matches that their deviations between the y pixels of the key were more than the DEVIATION_THRESHOLD.
    :param deviations:
    :param stereo_pair:
    :return:
    """
    stereo_pair.set_rectified_inliers_matches_idx(np.where(deviations <= DEVIATION_THRESHOLD)[0])

    left_image_rectified_inliers_kps, right_image_rectified_inliers_kps = [], []

    for rectified_inlier_match in stereo_pair.get_rectified_inliers_matches():
        left_image_rectified_inliers_kps.append(rectified_inlier_match.queryIdx)
        right_image_rectified_inliers_kps.append(rectified_inlier_match.trainIdx)

    stereo_pair.left_image.set_rectification_inliers_kps(left_image_rectified_inliers_kps)
    stereo_pair.right_image.set_rectification_inliers_kps(right_image_rectified_inliers_kps)

    # show_matches(stereo_pair.left_image, stereo_pair.right_image, stereo_pair.get_rectified_inliers_matches())
    # draw_good_and_bad_matches(stereo_pair, "rectified1", "rectified2", FilterMethod.RECTIFICATION)


def draw_good_and_bad_matches(stereo_pair: StereoPair, output_name1: str, output_name2: str, filter_method: FilterMethod) -> None:
    """
    This function plots inliers and outliers of the key points filtered by the filter method.
    :param stereo_pair:
    :param output_name1:
    :param output_name2:
    :param filter_method:
    :return:
    """
    img1, img2 = read_images(stereo_pair.idx, ImageColor.RGB)
    cv2.drawKeypoints(img1, stereo_pair.left_image.get_inliers_kps(filter_method), img1, (0, 128, 255))
    cv2.drawKeypoints(img1, stereo_pair.left_image.get_outliers_kps(filter_method), img1, (255, 255, 0))

    # cv2.drawKeypoints(img2, stereo_pair.right_image.get_inliers_kps(filter_method), img2, (0, 128, 255))
    # cv2.drawKeypoints(img2, stereo_pair.right_image.get_outliers_kps(filter_method), img2, (255, 255, 0))
    # cv2.imwrite(f"{output_name1}.png", img1)
    # cv2.imwrite(f"{output_name2}.png", img2)
    # cv2.imshow(output_name1, img1)
    # cv2.imshow(output_name2, img2)
    # cv2.waitKey(0)
    return img1


def read_cameras() -> Tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    """
    This function gets the R | t matrices of the ground truth cameras.
    :return:
    """
    with open(os.path.join(DATA_PATH, 'calib.txt')) as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def triangulate_point(P: FloatNDArray, Q: FloatNDArray, p_keypoint: KeyPoint, q_keypoint: KeyPoint) -> Tuple[FloatNDArray, np.float64]:
    """
    This function is triangulate a matched point from 2 key points from a stereo image.
    :param P:
    :param Q:
    :param p_keypoint:
    :param q_keypoint:
    :return:
    """
    A = np.array([P[2] * p_keypoint.pt[0] - P[0], P[2] * p_keypoint.pt[1] - P[1], Q[2] * q_keypoint.pt[0] - Q[0], Q[2] * q_keypoint.pt[1] - Q[1]])
    u, d, v_t = np.linalg.svd(A)
    our_p4d = v_t[-1]
    our_p3d = our_p4d[:3] / our_p4d[3]
    return our_p3d, our_p4d[3]


def is_our_triangulate_equal_cv(P: FloatNDArray, Q: FloatNDArray, p1: KeyPoint, p2: KeyPoint, cv_p3d: FloatNDArray) -> bool:
    """
    This function is a test to check whether our implentaion of a triangulation is the same as the opencv implementation.
    :param P:
    :param Q:
    :param p1:
    :param p2:
    :param cv_p3d:
    :return:
    """
    our_p3d, lamda = triangulate_point(P, Q, p1, p2)
    return np.all(np.isclose(our_p3d, cv_p3d))


def triangulate_all_points(matches: NDArray[DMatch], stereo_pair: StereoPair) -> FloatNDArray:
    """
    This function triangulates all the key points that matched in a stereo image.
    :param matches:
    :param stereo_pair:
    :return:
    """
    k, m1, m2 = read_cameras()
    P = k @ m1
    Q = k @ m2
    points = np.zeros((len(matches), 3))
    # equal = True
    for i, match in enumerate(matches):
        p1 = stereo_pair.left_image.kps[match.queryIdx]
        p2 = stereo_pair.right_image.kps[match.trainIdx]
        cv_p4d = cv2.triangulatePoints(P, Q, p1.pt, p2.pt).squeeze()
        cv_p3d = cv_p4d[:3] / cv_p4d[3]
        # equal = equal and is_our_triangulate_equal_cv(P, Q, p1, p2, cv_p3d)
        points[i] = cv_p3d

    # result_string = {False: "doesn't equal", True: "equals"}
    # print(f"Our triangulation {result_string[equal]} to cv triangulation")
    return points.T


def plot_triangulations(x: FloatNDArray, y: FloatNDArray, z: FloatNDArray) -> None:
    """
    This function plots the triangulations.
    :param x:
    :param y:
    :param z:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def compute_deltas(r, angle):
    m = np.tan(angle)
    dx = np.sqrt((r ** 2) / (m ** 2 + 1))
    dy = m * dx
    return dx, dy
def plot_trajectury(x: FloatNDArray, z: FloatNDArray, x2: FloatNDArray, z2: FloatNDArray, title: str = "traj", loop_locs: FloatNDArray = None,
                    real_y_angles=None, est_y_angles=None) -> None:
    """
    This function plots the trajectory we got compared to the real one.
    :param x:
    :param z:
    :param x2:
    :param z2:
    :param title:
    :return:
    """
    plt.scatter(x, z, c='blue', s=2, label='our trajectory')
    plt.scatter(x2, z2, c='red', s=2, label='ground truth location')
    if loop_locs is not None:
        plt.scatter(loop_locs.T[0], loop_locs.T[2], c='green', s=2, label="loop closure")
    plt.title("trajectory of left cameras and ground truth locations")
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.clf()


def plot_locations(x: FloatNDArray, z: FloatNDArray) -> None:
    """
    This function plots the locations of the cameras.
    :param x:
    :param z:
    :return:
    """
    plt.scatter(x, z, c='blue', s=2)
    # plt.xlim(x[0]-100, x[0]+100)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("trajectory of left cameras")
    plt.show()


def match_stereo_image(img_idx: int) -> StereoPair:
    """
    This function creates a StereoPair object which contains all the matches between 2 images after filtering by the stereo pattern.
    :param img_idx:
    :return:
    """
    img1, img2 = detect_key_points(img_idx)
    matches, img1_kps_to_matches_idx, img2_kps_to_matches_idx = match_key_points(img1, img2)
    stereo_pair = StereoPair(img1, img2, img_idx, matches)
    stereo_pair.set_images_kps_to_matches_idx(img1_kps_to_matches_idx, img2_kps_to_matches_idx)
    deviations = histogram_pattern(stereo_pair)
    pattern_reject_matches(deviations, stereo_pair)
    return stereo_pair


def match_pair_images_points(img_idx1: int, img_idx2: int, curr_stereo_pair1: Union[StereoPair, None] = None) -> Quad:
    """
    This function 2 pairs of stereo images and creates a Quad object that contains them both.
    :param img_idx1:
    :param img_idx2:
    :param curr_stereo_pair1:
    :return:
    """
    if curr_stereo_pair1 is None:
        stereo_pair1 = match_stereo_image(img_idx1)
    else:
        stereo_pair1 = curr_stereo_pair1
    stereo_pair2 = match_stereo_image(img_idx2)
    left_left_matches = match_key_points(stereo_pair1.left_image, stereo_pair2.left_image, False)[0]
    return Quad(stereo_pair1, stereo_pair2, left_left_matches)


def index_dict_matches(matches: NDArray[DMatch]) -> Dict[int, Tuple[int, int]]:
    """
    This function returns a dictionary that maps a index of the key point from the left image to pair of the
    index of the key point from right left image and the index of this specific match.
    :param matches:
    :return:
    """
    return {match.queryIdx: (match.trainIdx, i) for i, match in enumerate(matches)}


def create_quad(img_idx1: int, img_idx2: int, curr_stereo_pair2: StereoPair) -> Quad:
    """
    This function creates a quad object that includes the matches between img1 and img2.
    """
    quad = match_pair_images_points(img_idx1, img_idx2, curr_stereo_pair2)
    matches_1_dict = index_dict_matches(quad.stereo_pair1.get_rectified_inliers_matches())
    matches_2_dict = index_dict_matches(quad.stereo_pair2.get_rectified_inliers_matches())
    left_left_matches_dict = index_dict_matches(quad.left_left_matches)
    left_left_kps_idx_dict, left_right_img_kps_idx_dict1, left_right_img_kps_idx_dict2 = {}, {}, {}
    stereo_pair1_left_image_quad_inliers_kps_idx, stereo_pair2_left_image_quad_inliers_kps_idx = [], []
    stereo_pair1_quad_inliers_idx, stereo_pair2_quad_inliers_idx = [], []

    for left_img_pair1_kp_match, left_img_pair2_match in left_left_matches_dict.items():
        if left_img_pair1_kp_match in matches_1_dict and left_img_pair2_match[0] in matches_2_dict:
            stereo_pair1_left_image_quad_inliers_kps_idx.append(left_img_pair1_kp_match)
            stereo_pair2_left_image_quad_inliers_kps_idx.append(left_img_pair2_match[0])

            left_left_kps_idx_dict[left_img_pair2_match[0]] = left_img_pair1_kp_match
            left_right_img_kps_idx_dict1[left_img_pair1_kp_match] = matches_1_dict[left_img_pair1_kp_match][0]
            left_right_img_kps_idx_dict2[left_img_pair2_match[0]] = matches_2_dict[left_img_pair2_match[0]][0]

            stereo_pair1_quad_inliers_idx.append(matches_1_dict[left_img_pair1_kp_match][1])
            stereo_pair2_quad_inliers_idx.append(matches_2_dict[left_img_pair2_match[0]][1])

    quad.stereo_pair1.left_image.set_quad_inliers_kps_idx(stereo_pair1_left_image_quad_inliers_kps_idx)
    quad.stereo_pair2.left_image.set_quad_inliers_kps_idx(stereo_pair2_left_image_quad_inliers_kps_idx)

    quad.stereo_pair1.set_left_right_kps_idx_dict(left_right_img_kps_idx_dict1)
    quad.stereo_pair2.set_left_right_kps_idx_dict(left_right_img_kps_idx_dict2)

    quad.stereo_pair1.set_quad_inliers_matches_idx(stereo_pair1_quad_inliers_idx)
    quad.stereo_pair2.set_quad_inliers_matches_idx(stereo_pair2_quad_inliers_idx)

    quad.set_left_left_kps_idx_dict(left_left_kps_idx_dict)
    return quad


def find_matches_in_pair1(quad: Quad, pair2_left_img_kps_idx):
    """
    This function findS matches
    :param quad:
    :param pair2_left_img_kps_idx:
    :return:
    """
    pair1_left_img_kps_idx = [quad.left_left_kps_idx_dict[idx] for idx in pair2_left_img_kps_idx]
    matches_idx = quad.stereo_pair1.img1_kps_to_matches_idx[pair1_left_img_kps_idx]
    matches = quad.stereo_pair1.matches[matches_idx]
    return matches


def pnp_helper(quad: Quad, k: FloatNDArray, p3p: bool = True, indices: Union[NDArray[np.int32], None] = None) -> Tuple[bool, FloatNDArray, FloatNDArray]:
    """
    Helper function of pnp function.
    """
    flag = cv2.SOLVEPNP_EPNP
    kps_indices = indices
    if p3p:
        indices = np.random.choice(np.arange(len(quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.QUAD))), 4, replace=False)
        flag = cv2.SOLVEPNP_AP3P
        kps_indices = np.array(quad.stereo_pair2.left_image.get_inliers_kps_idx(FilterMethod.QUAD))[indices]
    good_kps = quad.stereo_pair2.left_image.kps[kps_indices]
    image_points = np.array([kps.pt for kps in good_kps])
    matches = find_matches_in_pair1(quad, kps_indices)
    points_3d = triangulate_all_points(matches, quad.stereo_pair1).T
    succeed, rvec, tvec = cv2.solvePnP(points_3d, image_points, k, None, flags=flag)
    return succeed, rvec, tvec


def pnp(quad: Quad, k: FloatNDArray, p3p: bool = True, inliers_idx: Union[NDArray[np.int32], None] = None) -> Tuple[FloatNDArray, FloatNDArray]:
    """
    This function perform PnP in order to get an R_t matrix.
    """
    succeed, rvec, tvec = pnp_helper(quad, k, p3p, inliers_idx)
    while not succeed:
        # print("didn't succeed")
        succeed, rvec, tvec = pnp_helper(quad, k, p3p, inliers_idx)
    R_t = rodriguez_to_mat(rvec, tvec)
    pair2_left_camera_location = transform_rt_to_location(R_t)
    return pair2_left_camera_location, R_t


def rodriguez_to_mat(rvec: FloatNDArray, tvec: FloatNDArray) -> FloatNDArray:
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def transform_rt_to_location(R_t: FloatNDArray, point_3d: Union[FloatNDArray, None] = None) -> FloatNDArray:
    """
    This function transform R_t matrix to (x,y,z) location.
    """
    R = R_t[:, :3]
    t = R_t[:, 3]
    if point_3d is None:
        point_3d = np.zeros(3)
    return R.T @ (point_3d - t)


def compute_camera_locations(img_idx1, img_idx2):
    k, m1, m2 = read_cameras()
    left_0_location = transform_rt_to_location(m1)[:, None]
    right_0_location = transform_rt_to_location(m2)[:, None]

    left_1_location = pnp(img_idx1, img_idx2, k)[0][:, None]
    right_1_location = (left_1_location + right_0_location)

    points = np.hstack((left_0_location, right_0_location, left_1_location, right_1_location))

    plot_triangulations(points[0], points[1], points[2])


def find_inliers(quad: Quad, k: FloatNDArray, current_transformation: FloatNDArray) -> Tuple[int, int, NDArray[np.int64]]:
    """
    This function finds the inliers by project 3d points using current_transformation and checking the pixel distance
    from the data we have about the point.
    """
    points_3d = triangulate_all_points(quad.stereo_pair1.get_quad_inliers_matches(), quad.stereo_pair1)
    points_4d = np.vstack((points_3d, np.ones(points_3d.shape[1])))
    model_pixels_2d = perform_transformation_3d_points_to_pixels(current_transformation, k, points_4d)
    real_pixels_2d = np.array([point.pt for point in quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.QUAD)]).T
    diff_real_and_model = np.abs(real_pixels_2d - model_pixels_2d)
    inliers_idx = np.where((diff_real_and_model[0] < PNP_THRESHOLD) & (diff_real_and_model[1] < PNP_THRESHOLD))[0]
    return len(inliers_idx), diff_real_and_model.shape[1] - len(inliers_idx), inliers_idx


def perform_transformation_3d_points_to_pixels(R_t_1_2: FloatNDArray, k: FloatNDArray, points_4d: FloatNDArray) -> FloatNDArray:
    """
    This function project 3d point using R_t_1_2 in order to find the point pixel.
    """
    pixels_3d = k @ R_t_1_2 @ points_4d
    pixels_3d[0] /= pixels_3d[2]
    pixels_3d[1] /= pixels_3d[2]
    model_pixels_2d = pixels_3d[:2]
    return model_pixels_2d


def compute_num_of_iter(p: float, epsilon: float, s: int) -> np.float64:
    """
    This function calculate the number of iteration.
    """
    return np.log(1 - p) / np.log(1 - ((1 - epsilon) ** s))


def ransac_helper(quad: Quad, k: FloatNDArray, max_num_inliers: int, p3p: bool, p: float, s: int, num_iter: np.float64,
                  pnp_inliers: Union[None, NDArray[np.int64]] = None) -> Tuple[int, np.float64, bool]:
    """
    Helper function of ransac function.
    """
    pair2_left_camera_location, current_transformation = pnp(quad, k, p3p, pnp_inliers)
    current_num_inliers, current_num_outliers, current_inliers_idx = find_inliers(quad, k, current_transformation)
    if not p3p and np.allclose(current_transformation, quad.get_relative_trans()):
        return max_num_inliers, num_iter, True
    if current_num_inliers > max_num_inliers:
        quad.set_relative_trans(current_transformation)
        quad.stereo_pair2.left_image.set_pnp_inliers_kps_idx(current_inliers_idx)
        max_num_inliers = current_num_inliers
        new_epsilon = current_num_outliers / (current_num_inliers + current_num_outliers)
        num_iter = compute_num_of_iter(p, new_epsilon, s)
    return max_num_inliers, num_iter, False


def ransac(img_idx1: int, img_idx2: int, k: FloatNDArray, curr_stereo_pair2: StereoPair = None, quad: Quad = None) -> Tuple[Quad, int]:
    """
    This function perform ransac using PnP as thr iner model, in order to find the location of stereo pair.
    :param img_idx1:
    :param img_idx2:
    :param k:
    :param curr_stereo_pair2:
    :param quad:
    :return:
    """
    s = RANSAC_NUM_SAMPLES
    p = RANSAC_SUCCESS_PROB
    epsilon = 0.85
    num_iter = compute_num_of_iter(p, epsilon, s)
    max_num_inliers = 0
    if not quad:
        quad = create_quad(img_idx1, img_idx2, curr_stereo_pair2)
    # Repeat 1
    i = 0
    while i <= num_iter:
        if i == 355:
            break
        max_num_inliers, num_iter = ransac_helper(quad, k, max_num_inliers, True, p, s, num_iter)[:2]
        i += 1
    # Repeat 2
    if max_num_inliers < 4:
        return quad, max_num_inliers
    for j in range(10):
        max_num_inliers, num_iter, is_transformation_close = ransac_helper(quad, k, max_num_inliers, False, p, s, num_iter,
                                                                           quad.stereo_pair2.left_image.get_inliers_kps_idx(FilterMethod.PNP))
        if is_transformation_close:
            break

    return quad, max_num_inliers


def plot_3d_clouds(points_3d_pair2: FloatNDArray, points_3d_pair2_projected2: FloatNDArray) -> None:
    """
    This function plot 3d point cloud.
    :param points_3d_pair2:
    :param points_3d_pair2_projected2:
    :return:
    """
    fig = plt.figure()
    plt.suptitle("3D point clouds of pair 2 and pair 2 projected from pair 1")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d_pair2[0], points_3d_pair2[1], points_3d_pair2[2], c="red", alpha=0.4)
    ax.scatter(points_3d_pair2_projected2[0], points_3d_pair2_projected2[1], points_3d_pair2_projected2[2], c="blue",
               alpha=0.4)
    ax.legend(["pair 2 3D point cloud", "pair 2 projected from pair 1"], loc='upper left')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def compute_2_3d_clouds(transformation: FloatNDArray, quad: Quad) -> Tuple[FloatNDArray, FloatNDArray]:
    """
    This function compute 3d point clouds
    :param transformation:
    :param quad:
    :return:
    """
    points_3d_pair2 = triangulate_all_points(quad.stereo_pair2.get_quad_inliers_matches(), quad.stereo_pair2)

    points_3d_pair1 = triangulate_all_points(quad.stereo_pair1.get_quad_inliers_matches(), quad.stereo_pair1)

    points_4d = np.vstack((points_3d_pair1, np.ones(points_3d_pair1.shape[1])))
    points_3d_pair2_projected = transformation @ points_4d
    points_3d_pair2_projected2 = (points_3d_pair2_projected.T[
        (np.abs(points_3d_pair2_projected[0]) < 20) & (np.abs(points_3d_pair2_projected[2]) < 100) & (np.abs(points_3d_pair2_projected[1]) < 8)]).T
    points_3d_pair2 = (points_3d_pair2.T[(np.abs(points_3d_pair2[0]) < 20) & (np.abs(points_3d_pair2[2]) < 100) & (np.abs(points_3d_pair2[1]) < 8)]).T
    # plot_3d_clouds(points_3d_pair2, points_3d_pair2_projected2)

    return points_3d_pair2, points_3d_pair2_projected


def compute_extrinsic_matrix(transformation_0_to_i: FloatNDArray,
                             transformation_i_to_i_plus_1: FloatNDArray) -> FloatNDArray:
    """
    This function calculate the transformation from 0 to i+1.
    :param transformation_0_to_i:
    :param transformation_i_to_i_plus_1:
    :return:
    """
    R1 = transformation_0_to_i[:, :3]
    t1 = transformation_0_to_i[:, 3]
    R2 = transformation_i_to_i_plus_1[:, :3]
    t2 = transformation_i_to_i_plus_1[:, 3]

    new_R = R2 @ R1
    new_t = R2 @ t1 + t2
    return np.hstack((new_R, new_t[:, None]))


def rotation_matrix_2_euler_angles(R):
    """
    This function extract the angles from a rotation matrix.
    :param R: a rotation matrix.
    :return: the angles: azimuth, pitch, roll.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def read_poses(first_index: int = 0, last_index: int = 3450) -> FloatNDArray:
    """
    This function reads the ground truth R_t and transform it to locations.
    :param first_index:
    :param last_index:
    :return:
    """
    locations = np.zeros((last_index - first_index, 3))
    i = 0
    with open(os.path.join(POSES_PATH, '00.txt')) as f:
        for l in f.readlines()[first_index:last_index]:
            # if i >= 500:  # for debug
            #     break
            l = l.split()
            extrinsic_matrix = np.array([float(i) for i in l])
            extrinsic_matrix = extrinsic_matrix.reshape(3, 4)

            ground_truth_loc = transform_rt_to_location(extrinsic_matrix)
            locations[i] = ground_truth_loc
            i += 1
    return locations


def trajectory() -> FloatNDArray:
    """
    This function create a trajectory after performing PnP (from the quad objects relative transformation).
    :return:
    """
    num_of_camerars = 3450
    k = read_cameras()[0]
    current_transformation = np.hstack((np.eye(3), np.zeros((3, 1))))
    locations = np.zeros((num_of_camerars, 3))
    curr_stereo_pair2 = None
    for i in range(num_of_camerars - 1):
        print("****************************************************************")
        print(i)
        print("****************************************************************")
        current_quad = ransac(i, i + 1, k, curr_stereo_pair2)[0]
        print(len(current_quad.stereo_pair1.left_image.kps))
        transformation_i_to_i_plus_1, curr_stereo_pair2 = current_quad.get_relative_trans(), current_quad.stereo_pair2
        transformation_0_to_i_plus_1 = compute_extrinsic_matrix(current_transformation, transformation_i_to_i_plus_1)
        locations[i + 1] = transform_rt_to_location(transformation_0_to_i_plus_1)
        # print(f"location of camera {i+1}: {locations[i + 1]}")
        current_transformation = transformation_0_to_i_plus_1
    return locations


def plot_local_error_traj(real_locs, est_locs, title):
    """
    This function plot the local error the norm between real_locs and est_locs.
    :param real_locs:
    :param est_locs:
    :param title:
    :return:
    """
    res = []
    dist_error = (real_locs - est_locs) ** 2
    error = np.sqrt(dist_error[0] + dist_error[1] + dist_error[2])
    for i in range(3450):
        res.append(error[i])
    plt.plot(res)
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.clf()


if __name__ == '__main__':
    random.seed(1)
    # for i in range(3):
    #     num = random.randint(0, 3449)
    #     match_stereo_image(num)
    # real_locs = read_poses().T
    est_locs = trajectory().T
    # plot_local_error_traj(real_locs, est_locs, "Initial_Trajectory_Error")
