import cv2
import numpy as np
import matplotlib.pyplot as plt

GRAY = 0
RGB = 1
DEVIATION_THRESHOLD = 2

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
    print("The percentage of matches that devaite by more than 2 pixel:",
          round(100 * (len(deviations[deviations > DEVIATION_THRESHOLD]) / len(
              deviations)), 2))
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
                              kps2, good_keypoint2, bad_keypoint2):
    cv2.drawKeypoints(img1, kps1[good_keypoint1], img1, (0, 128, 255))
    cv2.drawKeypoints(img1, kps1[bad_keypoint1], img1, (255, 255, 0))

    cv2.drawKeypoints(img2, kps2[good_keypoint2], img2, (0, 128, 255))
    cv2.drawKeypoints(img2, kps2[bad_keypoint2], img2, (255, 255, 0))
    # cv2.imwrite("pattern_output1.png", img1)
    # cv2.imwrite("pattern_output2.png", img2)
    cv2.imshow("Output pattern1", img1)
    cv2.imshow("Output pattern2", img2)
    cv2.waitKey(0)


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


def traingulate_all_points(matches, kps1, kps2):
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
        if our_p3d[2] < 0:
            print(lamda * our_p3d[0], lamda * our_p3d[1], lamda * our_p3d[2],
                  lamda)
            cv2.drawMatches(img1, kps1, img2, kps2, [match], res,
                        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite("outlier_match2.png", res)
        equal = equal and compare_to_cv_triangulation(P, Q, p1, p2, our_p3d)
        points[i] = our_p3d

    result_string = {False: "doesn't equal", True: "equals"}
    print(f"Our triangulation {result_string[equal]} to cv triangulation")
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


if __name__ == '__main__':
    des1, des2, img1, kps1, img2, kps2 = detect_key_points(0)
    # print_feature_descriptors(des1, des2)
    matches = match_key_points(des1, des2, img1, kps1, img2, kps2)
    deviations = histogram_pattern(matches, kps1, kps2)
    # significance_test(des1, des2, img1, kps1, img2, kps2)
    good_matches = pattern_reject_matches(0, deviations, kps1, kps2,
                                          matches)
    points = traingulate_all_points(good_matches, kps1, kps2)
    points = (
        points.T[(np.abs(points[0]) < 25) & (np.abs(points[2]) < 100)]).T
    plot_triangulations(points[0], points[1], points[2])
