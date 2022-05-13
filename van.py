import cv2
from image_utils import *
import pickle
import random
from typing import Tuple, Iterator
import os
import gtsam

DEVIATION_THRESHOLD = 2
RANSAC_NUM_SAMPLES = 4
RANSAC_SUCCESS_PROB = 0.99

DATA_PATH = os.path.join("VAN_ex","dataset","sequences", "00")
POSES_PATH = os.path.join("VAN_ex","dataset","poses")








def read_images(idx: int, color: ImageColor) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(os.path.join(DATA_PATH,'image_0', img_name), color)
    img2 = cv2.imread(os.path.join(DATA_PATH,'image_1', img_name), color)
    return img1, img2


def show_key_points(idx: int, kps1: NDArray[KeyPoint], kps2: NDArray[KeyPoint]) -> None:
    img1, img2 = read_images(idx, ImageColor.RGB)
    cv2.drawKeypoints(img1, kps1[:500], img1, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imshow("Output Image 1", img1)
    cv2.drawKeypoints(img2, kps2[:500], img2, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imshow("Output Image 2", img1)
    cv2.waitKey(0)


def detect_key_points(idx: int) -> Tuple[Image, Image]:
    img1_mat, img2_mat = read_images(idx, ImageColor.GRAY)
    detector = cv2.AKAZE_create()
    kps1, des1 = detector.detectAndCompute(img1_mat, None)
    kps2, des2 = detector.detectAndCompute(img2_mat, None)
    img1 = Image(img1_mat, np.array(kps1), des1)
    img2 = Image(img2_mat, np.array(kps2), des2)
    # show_key_points(idx, img1.kps , img2.kps)
    return img1, img2


def match_key_points(img1: Image, img2: Image) -> NDArray[DMatch]:
    brute_force = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = np.array(brute_force.match(img1.desc, img2.desc))
    # show_matches(img1, img2, matches)
    return matches


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
    # print("The percentage of matches that devaite by more than 2 pixel:",
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

    stereo_pair.left_image.set_inliers_kps(left_image_rectified_inliers_kps)
    stereo_pair.right_image.set_inliers_kps(right_image_rectified_inliers_kps)

    # draw_good_and_bad_matches(stereo_pair, "rectified1", "rectified2")


def draw_good_and_bad_matches(stereo_pair: StereoPair, output_name1: str, output_name2: str) -> None:
    img1, img2 = read_images(stereo_pair.idx, ImageColor.RGB)
    cv2.drawKeypoints(img1, stereo_pair.left_image.get_inliers_kps(FilterMethod.RECTIFICATION), img1, (0, 128, 255))
    cv2.drawKeypoints(img1, stereo_pair.left_image.get_rectification_outliers_kps(), img1, (255, 255, 0))

    cv2.drawKeypoints(img2, stereo_pair.right_image.get_inliers_kps(FilterMethod.RECTIFICATION), img2, (0, 128, 255))
    cv2.drawKeypoints(img2, stereo_pair.right_image.get_rectification_outliers_kps(), img2, (255, 255, 0))
    cv2.imwrite(f"{output_name1}.png", img1)
    cv2.imwrite(f"{output_name2}.png", img2)
    # cv2.imshow(output_name1, img1)
    # cv2.imshow(output_name2, img2)
    # cv2.waitKey(0)


def read_cameras() -> Tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    with open(os.path.join(DATA_PATH,'calib.txt')) as f:
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


def is_our_triangulate_equal_cv(P: FloatNDArray, Q: FloatNDArray, p1: KeyPoint, p2: KeyPoint, cv_p3d: FloatNDArray) -> bool:
    our_p3d, lamda = triangulate_point(P, Q, p1, p2)
    return np.all(np.isclose(our_p3d, cv_p3d))


def triangulate_all_points(matches: NDArray[DMatch], stereo_pair: StereoPair) -> FloatNDArray:
    k, m1, m2 = read_cameras()
    P = k @ m1
    Q = k @ m2
    points = np.zeros((len(matches), 3))
    equal = True
    for i, match in enumerate(matches):
        p1 = stereo_pair.left_image.kps[match.queryIdx]
        p2 = stereo_pair.right_image.kps[match.trainIdx]
        cv_p4d = cv2.triangulatePoints(P, Q, p1.pt, p2.pt).squeeze()
        cv_p3d = cv_p4d[:3] / cv_p4d[3]
        equal = equal and is_our_triangulate_equal_cv(P, Q, p1, p2, cv_p3d)
        points[i] = cv_p3d

    # result_string = {False: "doesn't equal", True: "equals"}
    # print(f"Our triangulation {result_string[equal]} to cv triangulation")
    return points.T


def triangulate_point(P: FloatNDArray, Q: FloatNDArray, p_keypoint: KeyPoint, q_keypoint: KeyPoint) -> Tuple[
    FloatNDArray, np.float64]:
    A = np.array([P[2] * p_keypoint.pt[0] - P[0],
                  P[2] * p_keypoint.pt[1] - P[1],
                  Q[2] * q_keypoint.pt[0] - Q[0],
                  Q[2] * q_keypoint.pt[1] - Q[1]])
    u, d, v_t = np.linalg.svd(A)
    our_p4d = v_t[-1]
    our_p3d = our_p4d[:3] / our_p4d[3]
    return our_p3d, our_p4d[3]


def plot_triangulations(x: FloatNDArray, y: FloatNDArray, z: FloatNDArray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def plot_trajectury(x: FloatNDArray, y: FloatNDArray, x2: FloatNDArray, y2: FloatNDArray) -> None:
    plt.scatter(x, y, c='blue', s=2)
    plt.scatter(x2, y2, c='red', s=2)
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.title("trajecory of left cameras and ground truth locations")
    plt.legend(('our trajectory', 'ground truth location'))
    plt.show()


def match_stereo_image(img_idx: int) -> StereoPair:
    img1, img2 = detect_key_points(img_idx)
    matches = match_key_points(img1, img2)
    stereo_pair = StereoPair(img1, img2, img_idx, matches)
    deviations = histogram_pattern(stereo_pair)
    pattern_reject_matches(deviations, stereo_pair)
    return stereo_pair


def match_pair_images_points(img_idx1: int, img_idx2: int, curr_stereo_pair1: Union[StereoPair, None] = None) -> Quad:
    if curr_stereo_pair1 is None:
        stereo_pair1 = match_stereo_image(img_idx1)
    else:
        stereo_pair1 = curr_stereo_pair1
    stereo_pair2 = match_stereo_image(img_idx2)
    left_left_matches = match_key_points(stereo_pair1.left_image, stereo_pair2.left_image)
    return Quad(stereo_pair1, stereo_pair2, left_left_matches)


def rodriguez_to_mat(rvec: FloatNDArray, tvec: FloatNDArray) -> FloatNDArray:
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def pnp_helper(quad: Quad, k: FloatNDArray, p3p: bool = True, indices: Union[NDArray[np.int32], None] = None) -> Tuple[bool, FloatNDArray, FloatNDArray]:
    flag = cv2.SOLVEPNP_EPNP
    if p3p:
        indices = np.random.choice(np.arange(len(quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.QUAD))), 4, replace=False)
        flag = cv2.SOLVEPNP_AP3P
    good_kps = quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.QUAD)[indices]
    image_points = np.array([kps.pt for kps in good_kps])
    points_3d = triangulate_all_points(quad.stereo_pair1.get_quad_inliers_matches()[indices], quad.stereo_pair1).T
    succeed, rvec, tvec = cv2.solvePnP(points_3d, image_points, k, None, flags=flag)
    return succeed, rvec, tvec


def pnp(quad: Quad, k: FloatNDArray, p3p: bool = True, inliers_idx: Union[NDArray[np.int32], None] = None) -> Tuple[FloatNDArray, FloatNDArray]:
    succeed, rvec, tvec = pnp_helper(quad, k, p3p, inliers_idx)
    while not succeed:
        print("didn't succeed")
        succeed, rvec, tvec = pnp_helper(quad, k, p3p, inliers_idx)
    R_t = rodriguez_to_mat(rvec, tvec)
    pair2_left_camera_location = transform_rt_to_location(R_t)
    return pair2_left_camera_location, R_t


def index_dict_matches(matches: NDArray[DMatch]) -> Dict[Tuple[int, int], int]:
    return {(match.queryIdx, match.trainIdx): i for i, match in enumerate(matches)}


def create_quad(img_idx1: int, img_idx2: int, curr_stereo_pair2: StereoPair) -> Quad:
    quad = match_pair_images_points(img_idx1, img_idx2, curr_stereo_pair2)
    # todo maybe reuse matches dict from last pair
    matches_1_dict = index_dict_matches(quad.stereo_pair1.get_rectified_inliers_matches())
    matches_2_dict = index_dict_matches(quad.stereo_pair2.get_rectified_inliers_matches())
    left_left_matches_dict = index_dict_matches(quad.left_left_matches)
    left_left_kps_idx_dict, left_right_img_kps_idx_dict1, left_right_img_kps_idx_dict2 = {}, {}, {}
    stereo_pair1_left_image_quad_inliers_kps_idx, stereo_pair2_left_image_quad_inliers_kps_idx = [], []
    stereo_pair1_quad_inliers_idx, stereo_pair2_quad_inliers_idx = [], []
    for match1 in matches_1_dict:
        for match2 in matches_2_dict:
            if (match1[0], match2[0]) in left_left_matches_dict:
                stereo_pair1_left_image_quad_inliers_kps_idx.append(match1[0])
                stereo_pair2_left_image_quad_inliers_kps_idx.append(match2[0])

                left_left_kps_idx_dict[match2[0]] = match1[0]
                left_right_img_kps_idx_dict1[match1[0]] = match1[1]
                left_right_img_kps_idx_dict2[match2[0]] = match2[1]

                stereo_pair1_quad_inliers_idx.append(matches_1_dict[match1])
                stereo_pair2_quad_inliers_idx.append(matches_2_dict[match2])
    quad.stereo_pair1.left_image.set_quad_inliers_kps_idx(stereo_pair1_left_image_quad_inliers_kps_idx)
    quad.stereo_pair2.left_image.set_quad_inliers_kps_idx(stereo_pair2_left_image_quad_inliers_kps_idx)

    quad.stereo_pair1.set_left_right_kps_idx_dict(left_right_img_kps_idx_dict1)
    quad.stereo_pair2.set_left_right_kps_idx_dict(left_right_img_kps_idx_dict2)

    quad.stereo_pair1.set_quad_inliers_matches_idx(stereo_pair1_quad_inliers_idx)
    quad.stereo_pair2.set_quad_inliers_matches_idx(stereo_pair2_quad_inliers_idx)

    quad.set_left_left_kps_idx_dict(left_left_kps_idx_dict)
    return quad


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
    inliers_idx = np.where((diff_real_and_model[0] < 2) & (diff_real_and_model[1] < 2))[0]
    return len(inliers_idx), diff_real_and_model.shape[1] - len(inliers_idx), inliers_idx


def perform_transformation_3d_points_to_pixels(R_t_1_2: FloatNDArray, k: FloatNDArray, points_4d: FloatNDArray) -> FloatNDArray:
    pixels_3d = k @ R_t_1_2 @ points_4d
    pixels_3d[0] /= pixels_3d[2]
    pixels_3d[1] /= pixels_3d[2]
    model_pixels_2d = pixels_3d[:2]
    return model_pixels_2d


def present_inliers_and_outliers(quad):
    # todo change this

    stereo_pair1_inliers_left_image_kps_idx = []
    for idx in quad.stereo_pair2.left_image.get_pnp_inliers_kps_idx():
        stereo_pair1_left_image_idx = quad.get_left_left_kps_idx_dict()[
            quad.stereo_pair2.left_image.get_quad_inliers_kps_idx()[idx]]
        stereo_pair1_inliers_left_image_kps_idx.append(stereo_pair1_left_image_idx)
    stereo_pair2_inliers_left_image_kps_idx = \
        quad.stereo_pair2.left_image.get_quad_inliers_kps_idx()[
            quad.stereo_pair2.left_image.get_pnp_inliers_kps_idx()]
    bad_keypoint1 = list(
        set(quad.stereo_pair1.left_image.get_quad_inliers_kps_idx()) - set(
            stereo_pair1_inliers_left_image_kps_idx))
    bad_keypoint2 = list(
        set(quad.stereo_pair2.left_image.get_quad_inliers_kps_idx()) - set(
            stereo_pair2_inliers_left_image_kps_idx))
    im1 = read_images(quad.stereo_pair1.idx, ImageColor.RGB)[0]
    im2 = read_images(quad.stereo_pair2.idx, ImageColor.RGB)[0]
    draw_good_and_bad_matches(im1, kps1_1,
                              stereo_pair1_inliers_left_image_kps_idx,
                              bad_keypoint1,
                              im2,
                              kps1_2, stereo_pair2_inliers_left_image_kps_idx,
                              bad_keypoint2, "left0",
                              "left1")


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


def ransac(img_idx1: int, img_idx2: int, k: FloatNDArray, curr_stereo_pair2: StereoPair) -> Quad:
    s = RANSAC_NUM_SAMPLES
    p = RANSAC_SUCCESS_PROB
    epsilon = 0.85
    num_iter = compute_num_of_iter(p, epsilon, s)
    max_num_inliers = 0
    quad = create_quad(img_idx1, img_idx2, curr_stereo_pair2)
    # Repeat 1
    i = 0
    while i <= num_iter:
        max_num_inliers, num_iter = ransac_helper(quad, k, max_num_inliers, True, p, s, num_iter)[:2]
        i += 1
    # Repeat 2
    for j in range(5):
        max_num_inliers, num_iter, is_transformation_close = ransac_helper(quad, k, max_num_inliers, False, p, s, num_iter,
                                                                           quad.stereo_pair2.left_image.get_inliers_kps_idx(FilterMethod.PNP))
        if is_transformation_close:
            break
    # if img_idx1 == 0:
    #     compute_2_3d_clouds(quad.get_relative_trans(), quad)
    # present_inliers_and_outliers(*best_compute_lst2)
    return quad


def plot_3d_clouds(points_3d_pair2: FloatNDArray, points_3d_pair2_projected2: FloatNDArray) -> None:
    fig = plt.figure()
    plt.suptitle("3D point clouds of pair 2 and pair 2 projected from pair 1")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d_pair2[0], points_3d_pair2[1], points_3d_pair2[2], c="red", alpha=0.4)
    ax.scatter(points_3d_pair2_projected2[0], points_3d_pair2_projected2[1], points_3d_pair2_projected2[2], c="blue", alpha=0.4)
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


def compute_extrinsic_matrix(transformation_0_to_i: FloatNDArray, transformation_i_to_i_plus_1: FloatNDArray) -> FloatNDArray:
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
            i += 1
    return locations


def trajectory() -> FloatNDArray:
    num_of_camerars = 30
    k = read_cameras()[0]
    current_transformation = np.hstack((np.eye(3), np.zeros((3, 1))))
    locations = np.zeros((num_of_camerars, 3))
    curr_stereo_pair2 = None
    for i in range(num_of_camerars - 1):
        print("****************************************************************")
        print(i)
        print("****************************************************************")
        current_quad = ransac(i, i + 1, k, curr_stereo_pair2)
        transformation_i_to_i_plus_1, curr_stereo_pair2 = current_quad.get_relative_trans(), current_quad.stereo_pair2
        transformation_0_to_i_plus_1 = compute_extrinsic_matrix(current_transformation, transformation_i_to_i_plus_1)
        locations[i + 1] = transform_rt_to_location(transformation_0_to_i_plus_1)
        current_transformation = transformation_0_to_i_plus_1
    return locations


# EX4 start

def create_quads(start_frame_id: int, end_frame_id: int) -> Iterator[Quad]:
    k = read_cameras()[0]
    curr_stereo_pair2 = None
    for i in range(start_frame_id, end_frame_id):
        print(i)
        current_quad = ransac(i, i + 1, k, curr_stereo_pair2)
        yield current_quad
        curr_stereo_pair2 = current_quad.stereo_pair2


def gen_track_id(index: int) -> Iterator[int]:
    while True:
        yield index
        index += 1


def get_idx_in_kp_left_image_pair2(quad: Quad, idx: int) -> int:
    return quad.stereo_pair2.left_image.get_quad_inliers_kps_idx()[idx]


def get_left_image_pair1_kps_idx(quad: Quad, idx: int) -> int:
    return quad.left_left_kps_idx_dict[quad.stereo_pair2.left_image.get_quad_inliers_kps_idx()[idx]]


def get_right_image_pair_kps_idx(pair: StereoPair, pair1_left_img_kp_idx: int) -> int:
    return pair.get_left_right_kps_idx_dict()[pair1_left_img_kp_idx]


def create_track(latest_tracks: List[Track], frames: List[Frame], inliers_idx: NDArray[np.int64], quad: Quad, track_id_gen: Iterator[int]) -> None:
    """
    This function create new track or continuing a track.
    :param latest_tracks: all tracks that created.
    :param frames: all frames that created.
    :param inliers_idx: the inliers of the left frame in pair1 of quad.
    :param quad:
    :param track_id_gen:
    """
    new_kps = {get_left_image_pair1_kps_idx(quad, idx) for idx in inliers_idx}
    kp_to_track_dict = {latest_tracks[track_id].last_kp_idx: track_id for track_id in frames[-2].track_ids}
    kp_to_track_dict = {kp_idx: kp_to_track_dict[kp_idx] for kp_idx in new_kps if kp_idx in kp_to_track_dict}
    for idx in inliers_idx:
        track_index = -1
        if get_left_image_pair1_kps_idx(quad, idx) in kp_to_track_dict:
            track_id = kp_to_track_dict[get_left_image_pair1_kps_idx(quad, idx)]
            track_index = kp_to_track_dict[get_left_image_pair1_kps_idx(quad, idx)]
        else:
            frame_id = quad.stereo_pair1.idx
            track_id = next(track_id_gen)
            frames[frame_id].track_ids.append(track_id)
            new_track = Track(track_id, get_idx_in_kp_left_image_pair2(quad, idx), quad.stereo_pair1.idx)
            latest_tracks.append(new_track)
            pair1_left_img_kp_idx = get_left_image_pair1_kps_idx(quad, idx)
            pair1_right_img_kp_idx = get_right_image_pair_kps_idx(quad.stereo_pair1, pair1_left_img_kp_idx)
            kp_l = quad.stereo_pair1.left_image.kps[pair1_left_img_kp_idx]
            x_l = kp_l.pt[0]
            x_r = quad.stereo_pair1.right_image.kps[pair1_right_img_kp_idx].pt[0]
            y = kp_l.pt[1]
            latest_tracks[-1].track_instances.append(TrackInstance(x_l, x_r, y))
            latest_tracks[-1].frame_ids.append(frame_id)

        right_frame_id = quad.stereo_pair2.idx
        frames[right_frame_id].track_ids.append(track_id)
        kp_l = quad.stereo_pair2.left_image.kps[get_idx_in_kp_left_image_pair2(quad, idx)]
        right_x_l = kp_l.pt[0]
        right_x_r = quad.stereo_pair2.right_image.kps[get_right_image_pair_kps_idx(quad.stereo_pair2, get_idx_in_kp_left_image_pair2(quad, idx))].pt[0]
        right_y = kp_l.pt[1]

        latest_tracks[track_index].set_last_pair_id(right_frame_id)
        latest_tracks[track_index].set_last_kp_idx(get_idx_in_kp_left_image_pair2(quad, idx))
        latest_tracks[track_index].track_instances.append(TrackInstance(right_x_l, right_x_r, right_y))
        latest_tracks[track_index].frame_ids.append(right_frame_id)


def create_database(start_frame_id: int, end_frame_id: int, start_track_id: int,
                    tracks: Union[List[Track], None] = None, frames: Union[List[Frame], None] = None) -> DataBase:
    """
    This function creates a database of tracks.
    :param start_frame_id: first frame to store the tracks from.
    :param end_frame_id: last frame to store the tracks.
    :param start_track_id: track id to start from.
    :param tracks:
    :param frames:
    :return: The database
    """

    if not tracks:
        tracks = []
    if not frames:
        frames = [Frame(start_frame_id)]
        frames[-1].set_transformation_from_zero(np.hstack((np.eye(3), np.zeros((3, 1)))))
    gen = gen_track_id(start_track_id)
    for i, quad in enumerate(create_quads(start_frame_id, end_frame_id)):
        current_transformation = frames[-1].get_transformation_from_zero()
        percentage = len(quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.PNP)) / len(quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.QUAD))
        frames[-1].set_inliers_percentage(round(percentage, 2))
        transformation_i_to_i_plus_1 = quad.get_relative_trans()
        transformation_0_to_i_plus_1 = compute_extrinsic_matrix(current_transformation, transformation_i_to_i_plus_1)
        frames.append(Frame(quad.stereo_pair2.idx))
        frames[-1].set_transformation_from_zero(transformation_0_to_i_plus_1)
        create_track(tracks, frames, quad.stereo_pair2.left_image.get_inliers_kps_idx(FilterMethod.PNP), quad, gen)
    percentage = len(quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.PNP)) / len(quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.QUAD))
    frames[-1].set_inliers_percentage(round(percentage, 2))
    return DataBase(tracks, frames)


def extend_database(database: DataBase, end_frame_id: int) -> DataBase:
    """
    This function extend the database from the last entry to end_frame_id.
    """
    start_frame_id = len(database.frames) - 1
    start_track_id_gen = len(database.tracks)
    new_database = create_database(start_frame_id, end_frame_id, start_track_id_gen, database.tracks, database.frames)
    return new_database


def save_database(database: DataBase) -> None:
    """
    This function save the database to a file.
    """
    with open("database.db", "wb") as file:
        pickle.dump(database, file)


def open_database() -> DataBase:
    """
    This function open the database file and return thr database object.
    """
    with open("database.db", "rb") as file:
        database = pickle.load(file)
    return database


def get_all_track_ids(frame_id: int, database: DataBase) -> List[int]:
    """
    This function return all the track_id that appear on a given frame_id.
    """
    return database.frames[frame_id].track_ids


def get_all_frame_ids(track_id: int, database: DataBase) -> List[int]:
    """
    This function return all the frame_id that are part of a given track_id.
    """
    return database.tracks[track_id].frame_ids


def get_feature_locations(frame_id: int, track_id: int, database: DataBase) -> TrackInstance:
    """
    This function return Feature locations of track TrackId on both left and right images as a triplet (x_l, x_r, y).
    """
    for frame, track_instance in zip(database.tracks[track_id].frame_ids, database.tracks[track_id].track_instances):
        if frame == frame_id:
            return track_instance


def result_image_size(image_shape: Tuple[int, int, int], x: int, y: int) -> Tuple[int, int, int, int]:
    """
    This function find the image size to present track
    :param image_shape:
    :param x: x track location
    :param y: y track location
    :return: The image size to display
    """
    x1 = max(0, x - 50) if image_shape[1] - x >= 50 else image_shape[1] - 100
    x2 = x1 + 100
    y1 = max(0, y - 50) if image_shape[0] - y >= 50 else image_shape[0] - 100
    y2 = y1 + 100
    return x1, x2, y1, y2


def display_track(database: DataBase) -> None:
    """
    This function display a random track of length > 9.
    :param database:
    """
    tracks_bigger_than_10 = list(np.array(database.tracks)[np.array(database.tracks) > 9])
    random_track = random.choice(tracks_bigger_than_10)
    frames_idx = get_all_frame_ids(random_track.track_id, database)
    for frame in frames_idx:
        track_instance = get_feature_locations(frame, random_track.track_id, database)
        img1, img2 = read_images(frame, ImageColor.RGB)
        kp1 = cv2.KeyPoint(track_instance.x_l, track_instance.y, 5)
        cv2.drawKeypoints(img1, [kp1], img1, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        kp2 = cv2.KeyPoint(track_instance.x_r, track_instance.y, 5)
        cv2.drawKeypoints(img2, [kp2], img2, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        res = np.empty((100, 200, 3), dtype=np.uint8)
        x1_l, x2_l, y1_l, y2_l = result_image_size(img1.shape, int(track_instance.x_l), int(track_instance.y))
        x1_r, x2_r, y1_r, y2_r = result_image_size(img2.shape, int(track_instance.x_r), int(track_instance.y))
        res[:, :100] = img1[y1_l:y2_l, x1_l:x2_l]
        res[:, 100:] = img2[y1_r:y2_r, x1_r:x2_r]
        cv2.imshow(f"{frame}", res)
        # cv2.imwrite(f"Output Image {frame}.png", res)
    cv2.waitKey(0)


def read_camera_matrices(first_index: int = 0, last_index: int = 3450) -> Iterator[FloatNDArray]:
    with open(os.path.join(POSES_PATH, '00.txt')) as f:
        for l in f.readlines()[first_index:last_index]:
            l = l.split()
            extrinsic_matrix = np.array([float(i) for i in l])
            extrinsic_matrix = extrinsic_matrix.reshape(3, 4)
            yield extrinsic_matrix


def calculate_norm(a: FloatNDArray, b: FloatNDArray) -> Union[float, FloatNDArray]:
    return np.linalg.norm(a - b)


def reprojection_error(extrinsic_matrix: FloatNDArray, k: FloatNDArray, p4d: FloatNDArray, location: List[float]) -> Union[float, FloatNDArray]:
    """
    :param extrinsic_matrix:
    :param k:
    :param p4d:
    :param location: [x,y]
    :return: reprojection error
    """
    projected_pixel_2d = perform_transformation_3d_points_to_pixels(extrinsic_matrix, k, p4d).squeeze()
    return calculate_norm(projected_pixel_2d, np.array(location))


def reprojection(database: DataBase) -> None:
    """
    This function calculate and plot the reprojection error (the distance between the projection and the tracked feature
     location on that camera)
    :param database: Tracks database
    """
    left_error = []
    right_error = []
    tracks_bigger_than_10 = list(np.array(database.tracks)[np.array(database.tracks) > 9])
    random_track = random.choice(tracks_bigger_than_10)
    k, m1, m2 = read_cameras()
    P = k @ m1
    Q = k @ m2
    track_location = random_track.track_instances[-1]
    cv_p4d = cv2.triangulatePoints(P, Q, (track_location.x_l, track_location.y), (track_location.x_r, track_location.y)).squeeze()
    cv_p3d = cv_p4d[:3] / cv_p4d[3]
    cv_p3d2 = transform_rt_to_location(next(read_camera_matrices(random_track.frame_ids[-1], random_track.frame_ids[-1] + 1)), cv_p3d)
    p4d = np.hstack((cv_p3d2, 1))[:, None]
    for i, extrinsic_matrix in enumerate(read_camera_matrices(random_track.frame_ids[0], random_track.frame_ids[-1] + 1)):
        location = random_track.track_instances[i]
        left_error.append(reprojection_error(extrinsic_matrix, k, p4d, [location.x_l, location.y]))

        right_extrinsic_matrix = compute_extrinsic_matrix(extrinsic_matrix, m2)
        right_error.append(reprojection_error(right_extrinsic_matrix, k, p4d, [location.x_r, location.y]))

    plt.plot(left_error, label="left error")
    plt.plot(right_error, label="right error")
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('error')
    plt.title('reprojection error')
    plt.show()


def present_statistics(database: DataBase) -> None:
    print("num_of_tracks: ", database.get_num_of_tracks())
    print("num_of_frames: ", database.get_num_of_frames())
    print("mean track length: ", database.get_mean_track_length())
    print("min track length: ", database.get_min_track_length())
    print("max track length: ", database.get_max_track_length())
    print("mean number of frame links: ", database.get_mean_number_of_frame_links())
    display_track(database)
    database.create_connectivity_graph()
    database.inliers_percentage_graph()
    database.create_track_length_histogram_graph()
    reprojection(database)


# EX4 end


# EX 5 start

def reprojection_error2(database: DataBase):
    tracks_bigger_than_10 = list(np.array(database.tracks)[np.array(database.tracks) > 9])
    random_track = random.choice(tracks_bigger_than_10)
    c =  gtsam.gtsam.StereoCamera()
if __name__ == '__main__':
    # database = create_database(0, 3449, 0)
    # compute_camera_locations(0, 1)
    # save_database(database)
    database = open_database()
    reprojection_error2(database)
    t=1
    # new_database = extend_database(database, 60)
    # save_database(new_database)
