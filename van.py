import cv2
import numpy as np

GRAY = 0
RGB = 1

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
    # cv2.imwrite("output1.png", img1)
    cv2.imshow("Output Image 1", img1)
    cv2.drawKeypoints(img1, kps2[:500], img2, (255, 0, 0),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    # cv2.imwrite("output2.png", img2)
    cv2.imshow("Output Image 2", img1)
    cv2.waitKey(0)


def detect_key_points(idx):
    img1, img2 = read_images(idx, GRAY)

    # TODO: maybe change it to a different detector
    detector = cv2.AKAZE_create()
    kps1, des1 = detector.detectAndCompute(img1, None)
    kps2, des2 = detector.detectAndCompute(img2, None)

    # show_key_points(idx, kps1, kps2)

    return des1, des2, img1, np.array(kps1), img2, np.array(kps2)


def match_key_points(des1, des2, img1, kps1, img2, kps2):
    res = np.empty(
        (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
        dtype=np.uint8)
    # TODO: maybe change it to a different distance method
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = np.array(brute_force.match(des1, des2))
    random_matches = matches[np.random.randint(len(matches), size=20)]
    cv2.drawMatches(img1, kps1, img2, kps2, random_matches, res,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Output matches", res)
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
    # TODO: maybe change it to a different distance method
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = brute_force.knnMatch(des1, des2, k=2)
    # TODO: maybe change it to a different ratio value
    ratio = 0.15
    good_matches = np.array(
        [m1 for m1, m2 in matches if m1.distance < ratio * m2.distance])
    random_matches = good_matches[
        np.random.randint(len(good_matches), size=20)]
    cv2.drawMatches(img1, kps1, img2, kps2, random_matches, res,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
                    )
    cv2.imshow("Output good matches", res)

    all_matches = np.array([m1 for m1, m2 in matches])[
        np.random.randint(len(good_matches), size=10)]
    cv2.drawMatches(img1, kps1, img2, kps2, all_matches, res,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
                    )
    cv2.imshow("Output all matches", res)
    cv2.waitKey(0)


if __name__ == '__main__':
    des1, des2, img1, kps1, img2, kps2 = detect_key_points(0)

    print_feature_descriptors(des1, des2)

    significance_test(des1, des2, img1, kps1, img2, kps2)
