import random

import cv2
from image_utils import *
from typing import Tuple
import os
from tqdm import tqdm

DEVIATION_THRESHOLD = 1
PNP_THRESHOLD = 2
RANSAC_NUM_SAMPLES = 4
RANSAC_SUCCESS_PROB = 0.99
MAHALANOBIS_DISTANCE_TEST = 500000
INLIERS_THRESHOLD = 100
CONSENSUS_MATCHING_THRESHOLD = 0.6

DATA_PATH = os.path.join("VAN_ex", "dataset", "sequences", "00")
POSES_PATH = os.path.join("VAN_ex", "dataset", "poses")


def read_images(idx: int, color: ImageColor) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(os.path.join(DATA_PATH, 'image_0', img_name), color)
    img2 = cv2.imread(os.path.join(DATA_PATH, 'image_1', img_name), color)
    return img1, img2


def show_key_points(idx: int, kps1: NDArray[KeyPoint], kps2: NDArray[KeyPoint]) -> None:
    img1, img2 = read_images(idx, ImageColor.RGB)
    cv2.drawKeypoints(img1, kps1[500:], img1, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imshow("Output Image 1", img1)
    cv2.drawKeypoints(img2, kps2[500:], img2, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imshow("Output Image 2", img2)
    cv2.waitKey(0)


def detect_key_points(idx: int) -> Tuple[Image, Image]:
    img1_mat, img2_mat = read_images(idx, ImageColor.GRAY)
    detector = cv2.AKAZE_create()
    kps1, des1 = detector.detectAndCompute(img1_mat, None)
    kps2, des2 = detector.detectAndCompute(img2_mat, None)
    img1 = Image(img1_mat, np.array(kps1), des1)
    img2 = Image(img2_mat, np.array(kps2), des2)
    # show_key_points(idx, img1.kps, img2.kps)
    return img1, img2


def match_key_points(img1: Image, img2: Image, set_matches_idx: bool = True) -> NDArray[DMatch]:
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
    random_matches = matches[np.random.randint(len(matches), size=20)]
    res = np.empty((max(img1.mat.shape[0], img2.mat.shape[0]), img1.mat.shape[1] + img2.mat.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1.mat, img1.kps, img2.mat, img2.kps, random_matches, res, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Output matches", res)  # 1.3
    cv2.waitKey(0)


def print_feature_descriptors(descriptors1: NDArray[np.uint8], descriptors2: NDArray[np.uint8]) -> None:
    print("The first two feature descriptors of image 1:")
    print(descriptors1[:2])
    print("The first two feature descriptors of image 2:")
    print(descriptors2[:2])


def significance_test(img1: Image, img2: Image) -> None:
    res = np.empty((max(img1.mat.shape[0], img2.mat.shape[0]), img1.mat.shape[1] + img2.mat.shape[1], 3),
                   dtype=np.uint8)
    brute_force = cv2.BFMatcher(cv2.NORM_L1)
    matches = brute_force.knnMatch(img1.desc, img2.desc, k=2)
    ratio = 0.5
    good_matches = np.array([m1 for m1, m2 in matches if m1.distance < ratio * m2.distance])
    random_matches = good_matches[np.random.randint(len(good_matches), size=20)]
    cv2.drawMatches(img1.mat, img1.kps, img2.mat, img2.kps, random_matches, res,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Output random good matches", res)  # 1.4
    cv2.waitKey(0)


def plot_histogram(deviations: FloatNDArray) -> None:
    plt.hist(deviations, 50)
    plt.title("Histogram of deviations between matches")
    plt.ylabel("Number of matches")
    plt.xlabel("Deviation from rectified stereo pattern")
    plt.show()


def histogram_pattern(stereo_pair: StereoPair) -> FloatNDArray:
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
    img1, img2 = read_images(stereo_pair.idx, ImageColor.RGB)
    # print(f"length of inliers kps img1: {len(stereo_pair.left_image.get_inliers_kps(filter_method))}")
    # print(f"length of outliers kps img1: {len(stereo_pair.left_image.get_outliers_kps(filter_method))}")
    # # print(f"length of inliers kps img2: {len(stereo_pair.right_image.get_inliers_kps(filter_method))}")
    # # print(f"length of outliers kps img2: {len(stereo_pair.right_image.get_outliers_kps(filter_method))}")
    # print(f"num of kps of image1: {len(stereo_pair.left_image.kps)}")
    # print(f"num of kps of image2: {len(stereo_pair.right_image.kps)}")
    # print(f"num of matches: {len(stereo_pair.matches)}")
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
    A = np.array([P[2] * p_keypoint.pt[0] - P[0], P[2] * p_keypoint.pt[1] - P[1], Q[2] * q_keypoint.pt[0] - Q[0], Q[2] * q_keypoint.pt[1] - Q[1]])
    u, d, v_t = np.linalg.svd(A)
    our_p4d = v_t[-1]
    our_p3d = our_p4d[:3] / our_p4d[3]
    return our_p3d, our_p4d[3]


def is_our_triangulate_equal_cv(P: FloatNDArray, Q: FloatNDArray, p1: KeyPoint, p2: KeyPoint, cv_p3d: FloatNDArray) -> bool:
    our_p3d, lamda = triangulate_point(P, Q, p1, p2)
    return np.all(np.isclose(our_p3d, cv_p3d))


def triangulate_all_points(matches: NDArray[DMatch], stereo_pair: StereoPair) -> FloatNDArray:
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def plot_trajectury(x: FloatNDArray, z: FloatNDArray, x2: FloatNDArray, z2: FloatNDArray, title: str = "traj") -> None:
    plt.scatter(x, z, c='blue', s=2)
    plt.scatter(x2, z2, c='red', s=2)
    # plt.xlabel("z")
    # plt.ylabel("y")
    plt.title("trajecory of left cameras and ground truth locations")
    plt.legend(('our trajectory', 'ground truth location'))
    plt.savefig(f"{title}.png")
    plt.clf()


def plot_locations(x: FloatNDArray, z: FloatNDArray) -> None:
    plt.scatter(x, z, c='blue', s=2)
    # plt.xlim(x[0]-100, x[0]+100)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("trajecory of left cameras")
    plt.show()


def match_stereo_image(img_idx: int) -> StereoPair:
    img1, img2 = detect_key_points(img_idx)
    matches, img1_kps_to_matches_idx, img2_kps_to_matches_idx = match_key_points(img1, img2)
    stereo_pair = StereoPair(img1, img2, img_idx, matches)
    stereo_pair.set_images_kps_to_matches_idx(img1_kps_to_matches_idx, img2_kps_to_matches_idx)
    deviations = histogram_pattern(stereo_pair)
    pattern_reject_matches(deviations, stereo_pair)
    return stereo_pair


def match_pair_images_points(img_idx1: int, img_idx2: int, curr_stereo_pair1: Union[StereoPair, None] = None) -> Quad:
    if curr_stereo_pair1 is None:
        stereo_pair1 = match_stereo_image(img_idx1)
    else:
        stereo_pair1 = curr_stereo_pair1
    stereo_pair2 = match_stereo_image(img_idx2)
    left_left_matches = match_key_points(stereo_pair1.left_image, stereo_pair2.left_image, False)[0]
    return Quad(stereo_pair1, stereo_pair2, left_left_matches)


# def index_dict_matches(matches: NDArray[DMatch]) -> Dict[int, Tuple[int, int]]:
#     return {(match.queryIdx, match.trainIdx):i for i, match in enumerate(matches)}

def index_dict_matches(matches: NDArray[DMatch]) -> Dict[int, Tuple[int, int]]:
    return {match.queryIdx: (match.trainIdx, i) for i, match in enumerate(matches)}


def create_quad(img_idx1: int, img_idx2: int, curr_stereo_pair2: StereoPair) -> Quad:
    quad = match_pair_images_points(img_idx1, img_idx2, curr_stereo_pair2)
    # todo maybe reuse matches dict from last pair
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
    # for match1 in matches_1_dict:
    #     for match2 in matches_2_dict:
    #         if (match1[0], match2[0]) in left_left_matches_dict:
    #             stereo_pair1_left_image_quad_inliers_kps_idx.append(match1[0])
    #             stereo_pair2_left_image_quad_inliers_kps_idx.append(match2[0])
    #
    #             left_left_kps_idx_dict[match2[0]] = match1[0]
    #             left_right_img_kps_idx_dict1[match1[0]] = match1[1]
    #             left_right_img_kps_idx_dict2[match2[0]] = match2[1]
    #
    #             stereo_pair1_quad_inliers_idx.append(matches_1_dict[match1])
    #             stereo_pair2_quad_inliers_idx.append(matches_2_dict[match2])

    quad.stereo_pair1.left_image.set_quad_inliers_kps_idx(stereo_pair1_left_image_quad_inliers_kps_idx)
    quad.stereo_pair2.left_image.set_quad_inliers_kps_idx(stereo_pair2_left_image_quad_inliers_kps_idx)

    quad.stereo_pair1.set_left_right_kps_idx_dict(left_right_img_kps_idx_dict1)
    quad.stereo_pair2.set_left_right_kps_idx_dict(left_right_img_kps_idx_dict2)

    quad.stereo_pair1.set_quad_inliers_matches_idx(stereo_pair1_quad_inliers_idx)
    quad.stereo_pair2.set_quad_inliers_matches_idx(stereo_pair2_quad_inliers_idx)

    quad.set_left_left_kps_idx_dict(left_left_kps_idx_dict)
    return quad


def find_matches_in_pair1(quad: Quad, pair2_left_img_kps_idx):
    pair1_left_img_kps_idx = [quad.left_left_kps_idx_dict[idx] for idx in pair2_left_img_kps_idx]
    matches_idx = quad.stereo_pair1.img1_kps_to_matches_idx[pair1_left_img_kps_idx]
    matches = quad.stereo_pair1.matches[matches_idx]
    return matches


def pnp_helper(quad: Quad, k: FloatNDArray, p3p: bool = True, indices: Union[NDArray[np.int32], None] = None) -> Tuple[bool, FloatNDArray, FloatNDArray]:
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


def pnp(quad: Quad, k: FloatNDArray, p3p: bool = True, inliers_idx: Union[NDArray[np.int32], None] = None) -> Tuple[
    FloatNDArray, FloatNDArray]:
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
    R = R_t[:, :3]
    t = R_t[:, 3]
    if point_3d is None:
        point_3d = np.zeros(3)
    return R.T @ (point_3d - t)


def compute_camera_locations(img_idx1, img_idx2):
    # TODO fix this
    k, m1, m2 = read_cameras()
    left_0_location = transform_rt_to_location(m1)[:, None]
    right_0_location = transform_rt_to_location(m2)[:, None]

    left_1_location = pnp(img_idx1, img_idx2, k)[0][:, None]
    right_1_location = (left_1_location + right_0_location)

    points = np.hstack((left_0_location, right_0_location, left_1_location, right_1_location))

    # print("left 0:",points.T[0])
    # print("right 0:", points.T[1])
    # print("left 1:", points.T[2])
    # print("right 1:", points.T[3])
    plot_triangulations(points[0], points[1], points[2])


def find_inliers(quad: Quad, k: FloatNDArray, current_transformation: FloatNDArray) -> Tuple[int, int, NDArray[np.int64]]:
    points_3d = triangulate_all_points(quad.stereo_pair1.get_quad_inliers_matches(), quad.stereo_pair1)
    points_4d = np.vstack((points_3d, np.ones(points_3d.shape[1])))
    model_pixels_2d = perform_transformation_3d_points_to_pixels(current_transformation, k, points_4d)
    real_pixels_2d = np.array([point.pt for point in quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.QUAD)]).T
    diff_real_and_model = np.abs(real_pixels_2d - model_pixels_2d)
    inliers_idx = np.where((diff_real_and_model[0] < PNP_THRESHOLD) & (diff_real_and_model[1] < PNP_THRESHOLD))[0]
    return len(inliers_idx), diff_real_and_model.shape[1] - len(inliers_idx), inliers_idx


def perform_transformation_3d_points_to_pixels(R_t_1_2: FloatNDArray, k: FloatNDArray,
                                               points_4d: FloatNDArray) -> FloatNDArray:
    pixels_3d = k @ R_t_1_2 @ points_4d
    pixels_3d[0] /= pixels_3d[2]
    pixels_3d[1] /= pixels_3d[2]
    model_pixels_2d = pixels_3d[:2]
    return model_pixels_2d


# def present_inliers_and_outliers(quad):
#     # todo change this
#
#     stereo_pair1_inliers_left_image_kps_idx = []
#     for idx in quad.stereo_pair2.left_image._get_pnp_inliers_kps_idx():
#         stereo_pair1_left_image_idx = quad.get_left_left_kps_idx_dict()[
#             quad.stereo_pair2.left_image._get_quad_inliers_kps_idx()[idx]]
#         stereo_pair1_inliers_left_image_kps_idx.append(stereo_pair1_left_image_idx)
#     stereo_pair2_inliers_left_image_kps_idx = \
#         quad.stereo_pair2.left_image._get_quad_inliers_kps_idx()[
#             quad.stereo_pair2.left_image._get_pnp_inliers_kps_idx()]
#     bad_keypoint1 = list(
#         set(quad.stereo_pair1.left_image._get_quad_inliers_kps_idx()) - set(
#             stereo_pair1_inliers_left_image_kps_idx))
#     bad_keypoint2 = list(
#         set(quad.stereo_pair2.left_image._get_quad_inliers_kps_idx()) - set(
#             stereo_pair2_inliers_left_image_kps_idx))
#     im1 = read_images(quad.stereo_pair1.idx, ImageColor.RGB)[0]
#     im2 = read_images(quad.stereo_pair2.idx, ImageColor.RGB)[0]
#     draw_good_and_bad_matches(im1, kps1_1,
#                               stereo_pair1_inliers_left_image_kps_idx,
#                               bad_keypoint1,
#                               im2,
#                               kps1_2, stereo_pair2_inliers_left_image_kps_idx,
#                               bad_keypoint2, "left0",
#                               "left1")


def compute_num_of_iter(p: float, epsilon: float, s: int) -> np.float64:
    return np.log(1 - p) / np.log(1 - ((1 - epsilon) ** s))


def ransac_helper(quad: Quad, k: FloatNDArray, max_num_inliers: int, p3p: bool, p: float, s: int, num_iter: np.float64,
                  pnp_inliers: Union[None, NDArray[np.int64]] = None) -> Tuple[int, np.float64, bool]:
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
        # print("first part: index and num_iter", i, num_iter)
        if i == 178:
            break
        max_num_inliers, num_iter = ransac_helper(quad, k, max_num_inliers, True, p, s, num_iter)[:2]
        i += 1
    # Repeat 2
    if max_num_inliers < 4:
        return quad, max_num_inliers
    for j in range(5):
        # print("secondb part: index and num_iter", j, num_iter)
        max_num_inliers, num_iter, is_transformation_close = ransac_helper(quad, k, max_num_inliers, False, p, s, num_iter,
                                                                           quad.stereo_pair2.left_image.get_inliers_kps_idx(FilterMethod.PNP))
        if is_transformation_close:
            break

    # immg1 = draw_good_and_bad_matches(quad.stereo_pair1, "1", "2", FilterMethod.PNP)
    # immg11 = draw_good_and_bad_matches(quad.stereo_pair1, "1", "2", FilterMethod.RECTIFICATION)
    # immg21 = draw_good_and_bad_matches(quad.stereo_pair2, "1", "2", FilterMethod.QUAD)
    # immg22 = draw_good_and_bad_matches(quad.stereo_pair2, "1", "2", FilterMethod.PNP)
    #
    # cv2.imshow("pair2 QUAD", immg21)
    # cv2.imshow("pair2 PNP", immg22)
    # cv2.waitKey(0)
    # if img_idx1 == 0:
    #     compute_2_3d_clouds(quad.get_relative_trans(), quad)
    # present_inliers_and_outliers(*best_compute_lst2)
    return quad, max_num_inliers


def plot_3d_clouds(points_3d_pair2: FloatNDArray, points_3d_pair2_projected2: FloatNDArray) -> None:
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
    R1 = transformation_0_to_i[:, :3]
    t1 = transformation_0_to_i[:, 3]
    R2 = transformation_i_to_i_plus_1[:, :3]
    t2 = transformation_i_to_i_plus_1[:, 3]

    new_R = R2 @ R1
    new_t = R2 @ t1 + t2
    return np.hstack((new_R, new_t[:, None]))


def read_poses(first_index: int = 0, last_index: int = 3450) -> FloatNDArray:
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
            # print(f"location of camera {i}: {locations[i]}")
            i += 1
    return locations


def trajectory() -> FloatNDArray:
    num_of_camerars = 3450
    k = read_cameras()[0]
    current_transformation = np.hstack((np.eye(3), np.zeros((3, 1))))
    locations = np.zeros((num_of_camerars, 3))
    curr_stereo_pair2 = None
    for i in tqdm(range(num_of_camerars - 1)):
        # print("****************************************************************")
        # print(i)
        # print("****************************************************************")
        current_quad = ransac(i, i + 1, k, curr_stereo_pair2)[0]
        transformation_i_to_i_plus_1, curr_stereo_pair2 = current_quad.get_relative_trans(), current_quad.stereo_pair2
        transformation_0_to_i_plus_1 = compute_extrinsic_matrix(current_transformation, transformation_i_to_i_plus_1)
        locations[i + 1] = transform_rt_to_location(transformation_0_to_i_plus_1)
        # print(f"location of camera {i+1}: {locations[i + 1]}")
        current_transformation = transformation_0_to_i_plus_1
    return locations


def plot_local_error_traj(real_locs, est_locs, title):
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
    est_locs = trajectory().T
    real_locs = read_poses().T
    plot_local_error_traj(real_locs, est_locs, "Initial_Trajectory_Error")