import cv2
import numpy as np
import matplotlib.pyplot as plt

GRAY = 0
RGB = 1
DEVIATION_THRESHOLD = 2
RANSAC_NUM_SAMPLES = 4
RANSAC_SUCCESS_PROB = 0.9

DATA_PATH = r'VAN_ex\dataset\sequences\00\\'


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
    img1, img2 = read_images(idx, GRAY)
    detector = cv2.AKAZE_create()
    kps1, des1 = detector.detectAndCompute(img1, None)
    kps2, des2 = detector.detectAndCompute(img2, None)
    # show_key_points(idx, kps1, kps2)
    return des1, des2, img1, np.array(kps1), img2, np.array(kps2)


def match_key_points(des1, des2, img1, kps1, img2, kps2):
    brute_force = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = np.array(brute_force.match(des1, des2))
    random_matches = matches[np.random.randint(len(matches), size=20)]
    # show_matches(img1, kps1, img2, kps2, random_matches)
    return matches


def show_matches(img1, kps1, img2, kps2, random_matches):
    res = np.empty(
        (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
        dtype=np.uint8)
    cv2.drawMatches(img1, kps1, img2, kps2, random_matches, res,
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


def histogram_pattern(matches, kps1, kps2):
    deviations = np.zeros(len(matches))
    for i, match in enumerate(matches):
        y1 = kps1[match.queryIdx].pt[1]
        y2 = kps2[match.trainIdx].pt[1]
        deviations[i] = abs(y2 - y1)
    # print("The percentage of matches that devaite by more than 2 pixel:",
    #       round(100 * (len(deviations[deviations > DEVIATION_THRESHOLD]) / len(
    #           deviations)), 2))
    # plot_histogram(deviations)
    return deviations


def pattern_reject_matches(img_idx, deviations, kps1, kps2, matches):
    img1, img2 = read_images(img_idx, RGB)
    good_matches = matches[deviations <= DEVIATION_THRESHOLD]
    bad_matches = matches[deviations > DEVIATION_THRESHOLD]
    good_keypoint1, good_keypoint2 = [], []
    bad_keypoint1, bad_keypoint2 = [], []
    for good_match in good_matches:
        good_keypoint1.append(good_match.queryIdx)
        good_keypoint2.append(good_match.trainIdx)
    for bad_match in bad_matches:
        bad_keypoint1.append(bad_match.queryIdx)
        bad_keypoint2.append(bad_match.trainIdx)

    # draw_good_and_bad_matches(img1, kps1, good_keypoint1, bad_keypoint1,
    # img2, kps2, good_keypoint2, bad_keypoint2)
    return good_matches


def draw_good_and_bad_matches(img1, kps1, good_keypoint1, bad_keypoint1, img2,
                              kps2, good_keypoint2, bad_keypoint2, output1_str, output2_str):
    cv2.drawKeypoints(img1, kps1[good_keypoint1], img1, (0, 128, 255))
    cv2.drawKeypoints(img1, kps1[bad_keypoint1], img1, (255, 255, 0))

    cv2.drawKeypoints(img2, kps2[good_keypoint2], img2, (0, 128, 255))
    cv2.drawKeypoints(img2, kps2[bad_keypoint2], img2, (255, 255, 0))
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


def triangulate_all_points(matches, kps1, kps2, img1, img2):
    k, m1, m2 = read_cameras()
    P = k @ m1
    Q = k @ m2
    points = np.zeros((len(matches), 3))
    equal = True
    res = np.empty(
        (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
        dtype=np.uint8)
    for i, match in enumerate(matches):
        p1 = kps1[match.queryIdx]
        p2 = kps2[match.trainIdx]
        our_p3d, lamda = triangulate_point(P, Q, p1, p2)
        # if our_p3d[2] < 0:
        #     print(lamda * our_p3d[0], lamda * our_p3d[1], lamda * our_p3d[2],
        #           lamda)
        #     cv2.drawMatches(img1, kps1, img2, kps2, [match], res,
        #                 flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        #     cv2.imwrite("outlier_match2.png", res)
        # equal = equal and compare_to_cv_triangulation(P, Q, p1, p2, our_p3d)
        points[i] = our_p3d

    result_string = {False: "doesn't equal", True: "equals"}
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

def plot_trajectury(x, y):
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def match_stereo_image(img_idx):
    des1, des2, img1, kps1, img2, kps2 = detect_key_points(img_idx)
    matches = match_key_points(des1, des2, img1, kps1, img2, kps2)
    deviations = histogram_pattern(matches, kps1, kps2)
    good_matches = pattern_reject_matches(img_idx, deviations, kps1, kps2,
                                          matches)
    # points = traingulate_all_points(good_matches, kps1, kps2, img1, img2)
    # points = (
    #     points.T[(np.abs(points[0]) < 25) & (np.abs(points[2]) < 100)]).T
    # plot_triangulations(points[0], points[1], points[2])
    return good_matches, kps1, kps2, des1, img1, img2


def match_pair_images_points(img_idx1, img_idx2):
    # $_& -> $ means the left or right image in stereo
    #        & means the index + 1 of the image
    matches_1, kps1_1, kps2_1, des1_1, img1_1, img2_1 = match_stereo_image(img_idx1)
    matches_2, kps1_2, kps2_2, des1_2, img1_2, img2_2 = match_stereo_image(img_idx2)
    left_left_matches = match_key_points(des1_1, des1_2, img1_1, kps1_1, img1_2, kps1_2)
    return matches_1, matches_2, left_left_matches, kps1_1, kps2_1,  kps1_2, kps2_2, img1_1, img2_1, img1_2

def index_dict_matches(matches):
    return {(match.queryIdx, match.trainIdx):i for i, match in enumerate(matches)}


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def pnp(img_idx1, img_idx2, k, p3p=True, inliers_idx=None):
    matches_1, matches_2, left_left_matches, kps1_1, kps2_1, kps1_2, kps2_2, img1_1, img2_1, img1_2 = match_pair_images_points(img_idx1, img_idx2)

    matches_1_dict, matches_2_dict, left_left_matches_dict = index_dict_matches(matches_1),index_dict_matches(matches_2),index_dict_matches(left_left_matches)
    kps1_2_matches_idx = []
    good_matches1_idx = []
    for match1 in matches_1_dict:
        for match2 in matches_2_dict:
            if (match1[0], match2[0]) in left_left_matches_dict:
                kps1_2_matches_idx.append(match2[0])
                good_matches1_idx.append(matches_1_dict[match1])


    indices = inliers_idx
    flag = cv2.SOLVEPNP_EPNP
    if p3p:
        indices = np.random.choice(np.arange(len(kps1_2[kps1_2_matches_idx])), 4, replace=False)
        flag = cv2.SOLVEPNP_AP3P


    good_kps = kps1_2[kps1_2_matches_idx][indices]
    image_points = np.array([kps.pt for kps in good_kps])

    points_3d = triangulate_all_points(matches_1[good_matches1_idx][indices], kps1_1, kps2_1, img1_1, img2_1).T

    _, rvec, tvec = cv2.solvePnP(points_3d, image_points, k, None, flags=flag)
    R_t_1_2 = rodriguez_to_mat(rvec, tvec)
    left_1_location = transform_rt_to_location(R_t_1_2)
    return left_1_location, R_t_1_2

def transform_rt_to_location(R_t):
    R = R_t[:, :3]
    t = R_t[:, 3]
    return R.T @ (-t)

def compute_camera_locations(img_idx1, img_idx2):
    k, m1, m2 = read_cameras()
    left_0_location = transform_rt_to_location(m1)[:,None]
    right_0_location = transform_rt_to_location(m2)[:,None]
    left_1_location = pnp(img_idx1, img_idx2)[0][:,None]
    right_1_location = (left_1_location + right_0_location)

    points = np.hstack((left_0_location, right_0_location, left_1_location, right_1_location))
    #
    # print("left 0:",points.T[0])
    # print("right 0:", points.T[1])
    # print("left 1:", points.T[2])
    # print("right 1:", points.T[3])
    plot_triangulations(points[0], points[1], points[2])


def find_inliers(img_idx1, img_idx2, R_t_1_2, k):
    matches_1, matches_2, left_left_matches, kps1_1, kps2_1, kps1_2, kps2_2, img1_1, img2_1, img1_2 = match_pair_images_points(img_idx1, img_idx2)
    # k = read_cameras()[0]
    matches_1_dict, matches_2_dict, left_left_matches_dict = index_dict_matches(matches_1),index_dict_matches(matches_2),index_dict_matches(left_left_matches)
    matches_kps1_dict = {}
    good_matches1_idx = []
    kps1_2_matches_idx, kps1_1_matches_idx = [], []
    for match1 in matches_1_dict:
        for match2 in matches_2_dict:
            if (match1[0], match2[0]) in left_left_matches_dict:
                kps1_2_matches_idx.append(match2[0])
                kps1_1_matches_idx.append(match1[0])
                matches_kps1_dict[match2[0]] = (match1[0], left_left_matches_dict[(match1[0], match2[0])])
                good_matches1_idx.append(matches_1_dict[match1])
    points_3d = triangulate_all_points(matches_1[good_matches1_idx], kps1_1, kps2_1, img1_1, img2_1)
    points_4d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

    # R_t_1_2 = pnp(img_idx1, img_idx2)[1]

    model_pixels_2d = perform_transformation_3d_points_to_pixels(R_t_1_2, k, points_4d)
    real_pixels_2d = np.array([point.pt for point in kps1_2[kps1_2_matches_idx]]).T
    diff_real_and_model = np.abs(real_pixels_2d - model_pixels_2d)
    inliers_idx = np.where((diff_real_and_model[0] < 2) & (diff_real_and_model[1] < 2))[0]

    # present_inliers_and_outliers(img_idx1, img_idx2, kps1_1,
    #                              kps1_1_matches_idx, kps1_2,
    #                              kps1_2_matches_idx, matches_kps1_dict,
    #                              inliers_idx)

    return len(inliers_idx), diff_real_and_model.shape[1] - len(inliers_idx), [img_idx1, img_idx2, kps1_1,
                             kps1_1_matches_idx, kps1_2,
                             kps1_2_matches_idx, matches_kps1_dict,
                             inliers_idx]


def perform_transformation_3d_points_to_pixels(R_t_1_2, k, points_4d):
    pixels_3d = k @ R_t_1_2 @ points_4d
    pixels_3d[0] /= pixels_3d[2]
    pixels_3d[1] /= pixels_3d[2]
    model_pixels_2d = pixels_3d[:2]
    return model_pixels_2d


def present_inliers_and_outliers(img_idx1, img_idx2, kps1_1,
                                 kps1_1_matches_idx, kps1_2,
                                 kps1_2_matches_idx, matches_kps1_dict,
                                 result2):
    best_matches_idxs = []
    best_kps1_1_idxs = []
    for idx in result2:
        kps1_1_idx, left_left_match_idx = matches_kps1_dict[kps1_2_matches_idx[idx]]
        best_matches_idxs.append(left_left_match_idx)
        best_kps1_1_idxs.append(kps1_1_idx)
    best_kps1_2_idxs = np.array(kps1_2_matches_idx)[result2]
    bad_keypoint1 = list(set(kps1_1_matches_idx) - set(best_kps1_1_idxs))
    bad_keypoint2 = list(set(kps1_2_matches_idx) - set(best_kps1_2_idxs))
    im1 = read_images(img_idx1, RGB)[0]
    im2 = read_images(img_idx2, RGB)[0]
    draw_good_and_bad_matches(im1, kps1_1, best_kps1_1_idxs, bad_keypoint1,
                              im2,
                              kps1_2, best_kps1_2_idxs, bad_keypoint2, "left0",
                              "left1")

def compute_num_of_iter(p, epsilon, s):
    return np.log(1-p) / np.log(1 - ((1 - epsilon)**s))
def ransac(img_idx1, img_idx2, k):
    s = RANSAC_NUM_SAMPLES
    p = RANSAC_SUCCESS_PROB
    epsilon = 0.85

    num_iter = compute_num_of_iter(p, epsilon, s)

    max_num_inliers = 0
    best_transformation = None
    best_compute_lst = []
    # Repeat 1
    i = 0
    while i <= num_iter:
        current_transformation = pnp(img_idx1, img_idx2, k, True)[1]
        current_num_inliers, current_num_outliers, current_compute_lst  = find_inliers(img_idx1, img_idx2, current_transformation, k)
        if current_num_inliers > max_num_inliers:
            best_transformation = current_transformation
            best_compute_lst = current_compute_lst
            max_num_inliers = current_num_inliers
            new_epsilon = current_num_outliers / (current_num_inliers + current_num_outliers)
            num_iter = compute_num_of_iter(p, new_epsilon, s)
        i+=1
    # Repeat 2
    best_transformation2 = best_transformation
    max_num_inliers2 = max_num_inliers
    best_compute_lst2 = best_compute_lst

    for j in range(3):
        current_transformation = pnp(img_idx1, img_idx2, k, False, best_compute_lst[-1])[1]
        if np.allclose(current_transformation, best_transformation2):
            break
        current_num_inliers, current_num_outliers, current_compute_lst = find_inliers(img_idx1, img_idx2,current_transformation, k)
        if current_num_inliers > max_num_inliers:
            best_transformation2 = current_transformation
            max_num_inliers = current_num_inliers
            best_compute_lst2 = current_compute_lst
    present_inliers_and_outliers(*best_compute_lst2)
    return best_transformation, best_compute_lst2

def compute_extrinsic_matrix(transformation_0_to_i, transformation_i_to_i_plus_1):
    R1 = transformation_0_to_i[:, :3]
    t1 = transformation_0_to_i[:, 3]
    R2 = transformation_i_to_i_plus_1[:, :3]
    t2 = transformation_i_to_i_plus_1[:, 3]

    new_R = R2 @ R1
    new_t = R2 @ t1 + t2
    return np.hstack((new_R, new_t[:,None]))


def trajectory():
    k = read_cameras()[0]
    current_transformation = np.hstack((np.eye(3), np.zeros((3,1))))
    locations = np.zeros((4, 3))
    for i in range(3):
        print(i)
        transformation_i_to_i_plus_1 = ransac(i, i+1, k)[0]
        transformation_0_to_i_plus_1 = compute_extrinsic_matrix(current_transformation, transformation_i_to_i_plus_1)
        locations[i+1] = transform_rt_to_location(transformation_0_to_i_plus_1)
        current_transformation = transformation_0_to_i_plus_1
    return locations
if __name__ == '__main__':
    # des1, des2, img1, kps1, img2, kps2 = detect_key_points(1)
    # # print_feature_descriptors(des1, des2)
    # matches = match_key_points(des1, des2, img1, kps1, img2, kps2)
    # deviations = histogram_pattern(matches, kps1, kps2)
    # # significance_test(des1, des2, img1, kps1, img2, kps2)
    # good_matches = pattern_reject_matches(0, deviations, kps1, kps2,
    #                                       matches)
    # points = traingulate_all_points(good_matches, kps1, kps2, img1, img2)
    # points = (
    #     points.T[(np.abs(points[0]) < 25) & (np.abs(points[2]) < 100)]).T
    # plot_triangulations(points[0], points[1], points[2])
    # print(trajectory())
    locations = trajectory().T
    plot_trajectury(locations[0], locations[2])