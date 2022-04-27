import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_utils import *

GRAY = 0
RGB = 1
DEVIATION_THRESHOLD = 2
RANSAC_NUM_SAMPLES = 4
RANSAC_SUCCESS_PROB = 0.99

DATA_PATH = r'VAN_ex\dataset\sequences\00\\'
POSES_PATH = r'VAN_ex\dataset\poses\\'


def read_images(idx, color):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + 'image_0\\' + img_name, color)
    img2 = cv2.imread(DATA_PATH + 'image_1\\' + img_name, color)
    return img1, img2


def show_key_points(idx, kps1, kps2):
    img1, img2 = read_images(idx, RGB)
    cv2.drawKeypoints(img1, kps1[:500], img1, (255, 0, 0),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imshow("Output Image 1", img1)
    cv2.drawKeypoints(img2, kps2[:500], img2, (255, 0, 0),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imshow("Output Image 2", img1)
    cv2.waitKey(0)


def detect_key_points(idx):
    img1_mat, img2_mat = read_images(idx, GRAY)
    detector = cv2.AKAZE_create()
    kps1, des1 = detector.detectAndCompute(img1_mat, None)
    kps2, des2 = detector.detectAndCompute(img2_mat, None)
    img1 = Image(img1_mat, np.array(kps1), des1)
    img2 = Image(img2_mat, np.array(kps2), des2)
    # show_key_points(idx, img1.kps , img2.kps)
    return img1, img2


def match_key_points(img1, img2):
    brute_force = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = np.array(brute_force.match(img1.desc, img2.desc))
    # show_matches(img1, img2, matches)
    return matches


def show_matches(img1, img2, matches):
    random_matches = matches[np.random.randint(len(matches), size=20)]
    res = np.empty(
        (max(img1.mat.shape[0], img2.mat.shape[0]),
         img1.mat.shape[1] + img2.mat.shape[1], 3),
        dtype=np.uint8)
    cv2.drawMatches(img1.mat, img1.kps, img2.mat, img2.kps, random_matches,
                    res,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Output matches", res)  # 1.3
    cv2.waitKey(0)


def print_feature_descriptors(descriptors1, descriptors2):
    print("The first two feature descriptors of image 1:")
    print(descriptors1[:2])
    print("The first two feature descriptors of image 2:")
    print(descriptors2[:2])


def significance_test(des1, des2, img1, kps1, img2, kps2):
    res = np.empty(
        (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
        dtype=np.uint8)
    brute_force = cv2.BFMatcher(cv2.NORM_L1)
    matches = brute_force.knnMatch(des1, des2, k=2)
    ratio = 0.5
    good_matches = np.array(
        [m1 for m1, m2 in matches if m1.distance < ratio * m2.distance])
    random_matches = good_matches[
        np.random.randint(len(good_matches), size=20)]
    cv2.drawMatches(img1, kps1, img2, kps2, random_matches, res,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Output random good matches", res)  # 1.4
    cv2.waitKey(0)


def plot_histogram(deviations):
    plt.hist(deviations, 50)
    plt.title("Histogram of deviations between matches")
    plt.ylabel("Number of matches")
    plt.xlabel("Deviation from rectified stereo pattern")
    plt.show()


def histogram_pattern(stereo_pair: StereoPair):
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


def pattern_reject_matches(deviations, stereo_pair):
    stereo_pair.set_rectified_inliers_matches_idx(
        np.where(deviations <= DEVIATION_THRESHOLD)[0])

    left_image_rectified_inliers_kps, right_image_rectified_inliers_kps = [], []

    for rectified_inlier_match in stereo_pair.get_rectified_inliers_matches():
        left_image_rectified_inliers_kps.append(
            rectified_inlier_match.queryIdx)
        right_image_rectified_inliers_kps.append(
            rectified_inlier_match.trainIdx)

    stereo_pair.left_image.set_inliers_kps(left_image_rectified_inliers_kps)
    stereo_pair.right_image.set_inliers_kps(right_image_rectified_inliers_kps)

    # draw_good_and_bad_matches(stereo_pair, "rectified1", "rectified2")


def draw_good_and_bad_matches(stereo_pair: StereoPair, output1_str, output2_str):
    img1, img2 = read_images(stereo_pair.idx, RGB)
    cv2.drawKeypoints(img1, stereo_pair.left_image.get_inliers_kps(FilterMethod.RECTIFICATION), img1,
                      (0, 128, 255))
    cv2.drawKeypoints(img1, stereo_pair.left_image.get_rectification_outliers_kps(), img1,
                      (255, 255, 0))

    cv2.drawKeypoints(img2, stereo_pair.right_image.get_inliers_kps(FilterMethod.RECTIFICATION), img2,
                      (0, 128, 255))
    cv2.drawKeypoints(img2, stereo_pair.right_image.get_rectification_outliers_kps(), img2,
                      (255, 255, 0))
    cv2.imwrite(f"{output1_str}.png", img1)
    cv2.imwrite(f"{output2_str}.png", img2)
    # cv2.imshow(output1_str, img1)
    # cv2.imshow(output2_str, img2)
    # cv2.waitKey(0)


def read_cameras():
    with open(DATA_PATH + 'calib.txt') as f:
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


def compare_to_cv_triangulation(P, Q, p_keypoint, q_keypoint, our_p3d):
    cv_p4d = cv2.triangulatePoints(P, Q, p_keypoint.pt,
                                   q_keypoint.pt).squeeze()
    cv_p3d = cv_p4d[:3] / cv_p4d[3]
    return np.all(np.isclose(our_p3d, cv_p3d))


def is_our_triangulate_equal_cv(P, Q, p1, p2):
    our_p3d, lamda = triangulate_point(P, Q, p1, p2)
    return compare_to_cv_triangulation(P, Q, p1, p2, our_p3d)


def triangulate_all_points(matches, stereo_pair):
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
        # equal = equal and is_our_triangulate_equal_cv(P, Q, p1, p2)
        points[i] = cv_p3d

    # result_string = {False: "doesn't equal", True: "equals"}
    # print(f"Our triangulation {result_string[equal]} to cv triangulation")
    return points.T


def triangulate_point(P, Q, p_keypoint, q_keypoint):
    A = np.array([P[2] * p_keypoint.pt[0] - P[0],
                  P[2] * p_keypoint.pt[1] - P[1],
                  Q[2] * q_keypoint.pt[0] - Q[0],
                  Q[2] * q_keypoint.pt[1] - Q[1]])
    u, d, v_t = np.linalg.svd(A)
    our_p4d = v_t[-1]
    our_p3d = our_p4d[:3] / our_p4d[3]
    return our_p3d, our_p4d[3]


def plot_triangulations(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def plot_trajectury(x, y, x2, y2):
    plt.scatter(x, y, c='blue', s=2)
    plt.scatter(x2, y2, c='red', s=2)
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.title("trajecory of left cameras and ground truth locations")
    plt.legend(('our trajectory', 'ground truth location'))
    plt.show()


def match_stereo_image(img_idx):
    img1, img2 = detect_key_points(img_idx)
    matches = match_key_points(img1, img2)
    stereo_pair = StereoPair(img1, img2, img_idx, matches)
    deviations = histogram_pattern(stereo_pair)
    pattern_reject_matches(deviations, stereo_pair)
    return stereo_pair


def match_pair_images_points(img_idx1, img_idx2, curr_stereo_pair1=None):
    if curr_stereo_pair1 is None:
        stereo_pair1 = match_stereo_image(img_idx1)
    else:
        stereo_pair1 = curr_stereo_pair1
    stereo_pair2 = match_stereo_image(img_idx2)
    left_left_matches = match_key_points(stereo_pair1.left_image,
                                         stereo_pair2.left_image)
    return Quad(stereo_pair1, stereo_pair2, left_left_matches)





def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def pnp_helper(quad, k, indices, p3p=True):
    flag = cv2.SOLVEPNP_EPNP
    if p3p:
        indices = np.random.choice(np.arange(len(quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.QUAD))), 4,
                                   replace=False)
        flag = cv2.SOLVEPNP_AP3P
    good_kps = quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.QUAD)[indices]
    image_points = np.array([kps.pt for kps in good_kps])
    points_3d = triangulate_all_points(
        quad.stereo_pair1.get_quad_inliers_matches()[indices],
        quad.stereo_pair1).T
    succeed, rvec, tvec = cv2.solvePnP(points_3d, image_points, k, None, flags=flag)
    return succeed, rvec, tvec


def pnp(k, p3p=True, inliers_idx=None, quad=None):
    succeed, rvec, tvec = pnp_helper(quad, k, inliers_idx, p3p)
    while not succeed:
        print("didn't succeed")
        succeed, rvec, tvec = pnp_helper(quad, k, inliers_idx, p3p)
    R_t = rodriguez_to_mat(rvec, tvec)
    pair2_left_camera_location = transform_rt_to_location(R_t)
    return pair2_left_camera_location, R_t


def index_dict_matches(matches):
    return {(match.queryIdx, match.trainIdx): i for i, match in enumerate(matches)}

def create_quad(img_idx1, img_idx2, curr_stereo_pair2):
    quad = match_pair_images_points(img_idx1, img_idx2, curr_stereo_pair2)
    # todo maybe reuse matches dict from last pair
    matches_1_dict, matches_2_dict, left_left_matches_dict = index_dict_matches(
        quad.stereo_pair1.get_rectified_inliers_matches()), index_dict_matches(
        quad.stereo_pair2.get_rectified_inliers_matches()), index_dict_matches(
        quad.left_left_matches)
    left_left_kps_idx_dict = {}
    left_right_img_kps_idx_dict1 = {}
    left_right_img_kps_idx_dict2 = {}
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
    quad.stereo_pair1.left_image.set_quad_inliers_kps_idx( stereo_pair1_left_image_quad_inliers_kps_idx)
    quad.stereo_pair2.left_image.set_quad_inliers_kps_idx(stereo_pair2_left_image_quad_inliers_kps_idx)

    quad.stereo_pair1.set_left_right_kps_idx_dict(left_right_img_kps_idx_dict1)
    quad.stereo_pair2.set_left_right_kps_idx_dict(left_right_img_kps_idx_dict2)

    quad.stereo_pair1.set_quad_inliers_matches_idx(stereo_pair1_quad_inliers_idx)
    quad.stereo_pair2.set_quad_inliers_matches_idx(stereo_pair2_quad_inliers_idx)

    quad.set_left_left_kps_idx_dict(left_left_kps_idx_dict)
    return quad


def transform_rt_to_location(R_t):
    R = R_t[:, :3]
    t = R_t[:, 3]
    return R.T @ (-t)


def compute_camera_locations(img_idx1, img_idx2):
    k, m1, m2 = read_cameras()
    left_0_location = transform_rt_to_location(m1)[:, None]
    right_0_location = transform_rt_to_location(m2)[:, None]
    left_1_location = pnp(img_idx1, img_idx2, k)[0][:, None]
    right_1_location = (left_1_location + right_0_location)

    points = np.hstack(
        (left_0_location, right_0_location, left_1_location, right_1_location))

    # print("left 0:",points.T[0])
    # print("right 0:", points.T[1])
    # print("left 1:", points.T[2])
    # print("right 1:", points.T[3])
    plot_triangulations(points[0], points[1], points[2])


def find_inliers(quad, k, current_transformation):
    points_3d = triangulate_all_points(
        quad.stereo_pair1.get_quad_inliers_matches(), quad.stereo_pair1)
    points_4d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

    model_pixels_2d = perform_transformation_3d_points_to_pixels(
        current_transformation, k, points_4d)
    real_pixels_2d = np.array([point.pt for point in
                               quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.QUAD)]).T
    diff_real_and_model = np.abs(real_pixels_2d - model_pixels_2d)
    inliers_idx = \
        np.where((diff_real_and_model[0] < 2) & (diff_real_and_model[1] < 2))[0]

    return len(inliers_idx), diff_real_and_model.shape[1] - len(
        inliers_idx), inliers_idx


def perform_transformation_3d_points_to_pixels(R_t_1_2, k, points_4d):
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
    im1 = read_images(quad.stereo_pair1.idx, RGB)[0]
    im2 = read_images(quad.stereo_pair2.idx, RGB)[0]
    draw_good_and_bad_matches(im1, kps1_1,
                              stereo_pair1_inliers_left_image_kps_idx,
                              bad_keypoint1,
                              im2,
                              kps1_2, stereo_pair2_inliers_left_image_kps_idx,
                              bad_keypoint2, "left0",
                              "left1")


def compute_num_of_iter(p, epsilon, s):
    return np.log(1 - p) / np.log(1 - ((1 - epsilon) ** s))


def ransac_helper(quad, k, max_num_inliers, p3p, p, s, num_iter, pnp_inliers=None):
    pair2_left_camera_location, current_transformation = pnp(k, p3p, pnp_inliers, quad)
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


def ransac(img_idx1, img_idx2, k, curr_stereo_pair2):
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
        max_num_inliers, num_iter, is_transformation_close = ransac_helper(quad, k, max_num_inliers, False, p, s,num_iter, quad.stereo_pair2.left_image.get_inliers_kps_idx(FilterMethod.PNP))
        if is_transformation_close:
            break
    # if img_idx1 == 0:
    #     compute_2_3d_clouds(quad.get_relative_trans(), quad)
    # present_inliers_and_outliers(*best_compute_lst2)
    return quad


def compute_2_3d_clouds(transformation, quad):
    points_3d_pair2 = triangulate_all_points(
        quad.stereo_pair2.get_quad_inliers_matches(), quad.stereo_pair2)

    points_3d_pair1 = triangulate_all_points(
        quad.stereo_pair1.get_quad_inliers_matches(), quad.stereo_pair1)

    points_4d = np.vstack((points_3d_pair1, np.ones(points_3d_pair1.shape[1])))
    points_3d_pair2_projected = (transformation @ points_4d)
    points_3d_pair2_projected2 = (points_3d_pair2_projected.T[
        (np.abs(points_3d_pair2_projected[0]) < 20) & (
                np.abs(points_3d_pair2_projected[2]) < 100) & (
                np.abs(points_3d_pair2_projected[1]) < 8)]).T

    points_3d_pair2 = (points_3d_pair2.T[(np.abs(points_3d_pair2[0]) < 20) & (
            np.abs(points_3d_pair2[2]) < 100) & (np.abs(
        points_3d_pair2[1]) < 8)]).T

    fig = plt.figure()
    plt.suptitle("3D point clouds of pair 2 and pair 2 projected from pair 1")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d_pair2[0], points_3d_pair2[1], points_3d_pair2[2],
               c="red", alpha=0.4)
    ax.scatter(points_3d_pair2_projected2[0], points_3d_pair2_projected2[1],
               points_3d_pair2_projected2[2], c="blue", alpha=0.4)
    ax.legend(["pair 2 3D point cloud", "pair 2 projected from pair 1"],
              loc='upper left')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    return points_3d_pair2, points_3d_pair2_projected


def compute_extrinsic_matrix(transformation_0_to_i,
                             transformation_i_to_i_plus_1):
    R1 = transformation_0_to_i[:, :3]
    t1 = transformation_0_to_i[:, 3]
    R2 = transformation_i_to_i_plus_1[:, :3]
    t2 = transformation_i_to_i_plus_1[:, 3]

    new_R = R2 @ R1
    new_t = R2 @ t1 + t2
    return np.hstack((new_R, new_t[:, None]))


def read_poses():
    locations = np.zeros((3450, 3))
    i = 0
    with open(POSES_PATH + '00.txt') as f:
        for l in f.readlines():
            # if i >= 500:  # for debug
            #     break
            l = l.split()
            extrinsic_matrix = np.array([float(i) for i in l])
            extrinsic_matrix = extrinsic_matrix.reshape(3, 4)
            ground_truth_loc = transform_rt_to_location(extrinsic_matrix)
            locations[i] = ground_truth_loc
            i += 1
    return locations






def trajectory():
    num_of_camerars = 3450
    k = read_cameras()[0]
    current_transformation = np.hstack((np.eye(3), np.zeros((3, 1))))
    locations = np.zeros((num_of_camerars, 3))
    curr_stereo_pair2 = None
    for i in range(num_of_camerars - 1):
        print(i)
        current_quad = ransac(i, i + 1, k, curr_stereo_pair2)
        yield current_quad
        transformation_i_to_i_plus_1, curr_stereo_pair2 = current_quad.get_relative_trans(), current_quad.stereo_pair2
        transformation_0_to_i_plus_1 = compute_extrinsic_matrix(current_transformation, transformation_i_to_i_plus_1)
        locations[i + 1] = transform_rt_to_location( transformation_0_to_i_plus_1)
        current_transformation = transformation_0_to_i_plus_1
    return locations


# EX4 start

def gen_track_id():
    i = 0
    while True:
        yield i
        i += 1


def create_track(latest_tracks, inliers_idx, quad: Quad, track_id_gen):
    frame_id = quad.stereo_pair1.idx
    for idx in inliers_idx:
        for track in latest_tracks:
            if track.last_pair == quad.stereo_pair1.idx-1 and  track.last_kp_idx == quad.stereo_pair2.left_image.get_quad_inliers_kps_idx()[idx]:
                track_id = track.track_id
                break
        else:
            track_id = next(track_id_gen)
            new_track = Track(track_id, quad.stereo_pair2.left_image.get_quad_inliers_kps_idx()[idx], quad.stereo_pair1.idx)
            latest_tracks.append(new_track)

        pair1_left_img_kp_idx = quad.left_left_kps_idx_dict[quad.stereo_pair2.left_image.get_quad_inliers_kps_idx()[idx]]
        pair1_right_img_kp_idx = quad.stereo_pair1.get_left_right_kps_idx_dict()[pair1_left_img_kp_idx]

        x_l = quad.stereo_pair1.left_image.kps[pair1_left_img_kp_idx].pt[0]
        x_r = quad.stereo_pair1.right_image.kps[pair1_right_img_kp_idx].pt[0]
        y = quad.stereo_pair1.left_image.kps[pair1_left_img_kp_idx].pt[1]
        yield track_id, frame_id, x_l, x_r, y




def create_database():
    latest_tracks = []
    gen = gen_track_id()
    for quad in trajectory():
        for track_id, frame_id, x_l, x_r, y in create_track(latest_tracks, quad.stereo_pair2.left_image.get_inliers_kps_idx(FilterMethod.PNP), quad, gen):
            print(track_id, frame_id, x_l, x_r, y)


# EX4 end


if __name__ == '__main__':
    create_database()
