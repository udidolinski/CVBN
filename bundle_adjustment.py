from database import *
from gtsam import gtsam, utils
from gtsam.gtsam import NonlinearFactorGraph, GenericStereoFactor3D
# from gtsam.gtsam import BetweenFactorPose3
from gtsam.noiseModel import Gaussian


def get_stereo_k() -> gtsam.Cal3_S2Stereo:
    """
    This function create and return calibration stereo camera.
    """
    k, m1, m2 = read_cameras()
    f_x, f_y, skew, c_x, c_y, baseline = k[0][0], k[1][1], k[0][1], k[0][2], k[1][2], m2[0][3]
    stereo_k = gtsam.Cal3_S2Stereo(f_x, f_y, skew, c_x, c_y, -baseline)
    return stereo_k


def get_camera_to_global(R_t: FloatNDArray) -> Tuple[FloatNDArray, FloatNDArray]:
    """
    This function convert given transformation: global->camera to camera->global.
    """
    R = R_t[:, :3]
    t = R_t[:, 3]
    new_R = R.T
    new_t = - new_R @ t
    return new_R, new_t[:, None]


def create_stereo_camera(database: DataBase, frame_idx: int, stereo_k: gtsam.Cal3_S2Stereo, start_frame_trans) -> Tuple[gtsam.StereoCamera, gtsam.Pose3]:
    """
    This function create and return stereo camera for a given frame.
    """
    curr_frame = database.frames[frame_idx]
    new_R, new_t = get_camera_to_global(curr_frame.transformation_from_zero)

    i_to_zero_trans = np.hstack((new_R, new_t))
    i_to_start_trans = compute_extrinsic_matrix(i_to_zero_trans, start_frame_trans)
    new_R = i_to_start_trans[:, :3]
    new_t = i_to_start_trans[:, 3]

    frame_pose = gtsam.Pose3(gtsam.Rot3(new_R), new_t)
    return gtsam.StereoCamera(frame_pose, stereo_k), frame_pose


def reprojection_error(database: DataBase):
    """
    This function calculate the reprojection error of all tracks in database.
    """
    tracks_bigger_than_10 = list(np.array(database.tracks)[np.array(database.tracks) > 9])
    random_track = random.choice(tracks_bigger_than_10)
    start_frame_trans = database.frames[random_track.frame_ids[0]].transformation_from_zero
    k, m1, m2 = read_cameras()
    f_x, f_y, skew, c_x, c_y, baseline = k[0][0], k[1][1], k[0][1], k[0][2], k[1][2], m2[0][3]
    stereo_k = gtsam.Cal3_S2Stereo(f_x, f_y, skew, c_x, c_y, -baseline)

    last_frame_stereo_camera, last_frame_pose = create_stereo_camera(database, random_track.frame_ids[-1], stereo_k, start_frame_trans)
    point_3d = last_frame_stereo_camera.backproject(gtsam.StereoPoint2(*random_track.track_instances[-1]))

    left_error, right_error = [], []
    graph = gtsam.NonlinearFactorGraph()
    x_last = gtsam.symbol('x', len(random_track.frame_ids))
    graph.add(gtsam.NonlinearEqualityPose3(x_last, last_frame_pose))
    stereo_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))
    l1 = gtsam.symbol('l', 1)  # point
    initialEstimate = gtsam.Values()
    initialEstimate.insert(x_last, last_frame_pose)
    factors = []
    for i, frame_idx in enumerate(random_track.frame_ids):
        if i == len(random_track.frame_ids) - 1:
            break
        frame_symbol = gtsam.symbol('x', i + 1)  # camera i
        curr_camera, frame_pose = create_stereo_camera(database, frame_idx, stereo_k, start_frame_trans)
        projected_p = curr_camera.project(point_3d)
        left_pt = np.array([projected_p.uL(), projected_p.v()])
        right_pt = np.array([projected_p.uR(), projected_p.v()])
        location = random_track.track_instances[i]
        left_location = np.array([location.x_l, location.y])
        right_location = np.array([location.x_r, location.y])
        left_error.append(calculate_norm(left_pt, left_location))
        right_error.append(calculate_norm(right_pt, right_location))
        factor = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(*location), stereo_model, frame_symbol, l1, stereo_k)
        graph.add(factor)
        factors.append(factor)
        initialEstimate.insert(frame_symbol, frame_pose)
    expected_l1 = point_3d
    initialEstimate.insert(l1, expected_l1)

    factor_errors = []
    for factor in factors:
        factor_errors.append(factor.error(initialEstimate))
    plot_projection_error('reprojection error', left_error, right_error)
    plot_projection_error('factor error', factor_errors)
    plot_reprojection_compared_to_factor_error(left_error, factor_errors)


def perform_bundle_window(database: DataBase, stereo_k: gtsam.Cal3_S2Stereo, bundle_frames: List[int]) -> Tuple[
    float, float, List[FloatNDArray], gtsam.NonlinearFactorGraph, gtsam.Values, List[int]]:
    """
    This function perform bundle adjustment optimization on a given frames (bundle_frames).
    """
    start_frame = bundle_frames[0]
    start_frame_trans = database.frames[start_frame].transformation_from_zero
    initialEstimate = gtsam.Values()
    graph = gtsam.NonlinearFactorGraph()
    x_start = gtsam.symbol('x', start_frame)
    stereo_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 0.5, 0.1]))
    graph.add(gtsam.PriorFactorPose3(x_start, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))))
    track_id_to_point = {}
    initialEstimate.insert(x_start, gtsam.Pose3())
    visited_tracks = set()
    frame_symbols = []
    factors = []
    to_remove_landmarks = set()
    for i in bundle_frames[::-1]:
        frame_symbol = gtsam.symbol('x', i) if i != start_frame else x_start  # camera i
        frame_symbols.append(frame_symbol)
        curr_camera = create_stereo_camera(database, i, stereo_k, start_frame_trans)[0]
        tracks = database.frames[i].track_ids
        tracks = [track_id for track_id in tracks if len(database.tracks[track_id].frame_ids) > 3]
        frame_pose = curr_camera.pose()
        if i != start_frame:
            initialEstimate.insert(frame_symbol, frame_pose)

        for track_id in tracks:
            location = get_feature_locations(i, track_id, database)
            if track_id not in visited_tracks:
                s = gtsam.symbol('l', track_id)  # feature point
                visited_tracks.add(track_id)
                point_3d = curr_camera.backproject(gtsam.StereoPoint2(*location))
                track_id_to_point[track_id] = (s, point_3d)
                landmark = s
                x = point_3d[0]
                z = point_3d[2]
                if landmark in to_remove_landmarks or abs(x) > 25 or z > 87 or z < 0:
                    to_remove_landmarks.add(landmark)
                    continue
                initialEstimate.insert(s, point_3d)
            landmark = gtsam.symbol('l', track_id)
            factor = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(*location), stereo_model, frame_symbol, track_id_to_point[track_id][0], stereo_k)
            if landmark in to_remove_landmarks:
                continue
            if factor.error(initialEstimate) > 5000:
                initialEstimate.erase(track_id_to_point[track_id][0])
                to_remove_landmarks.add(landmark)
                continue
            factors.append(factor)
            graph.add(factor)
    factor_num = graph.nrFactors()
    for i in range(1, factor_num):
        f = graph.at(i)
        landmark = f.keys()[1]
        if landmark in to_remove_landmarks:
            graph.remove(i)

    error_before = graph.error(initialEstimate)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
    result = optimizer.optimize()
    error_after = optimizer.error()
    last_frame_pose = result.atPose3(frame_symbols[0])
    print("first bundle total error before optimization: ", error_before)
    print("first bundle total error after optimization: ", error_after)
    return error_before, error_after, last_frame_pose, graph, result, frame_symbols



def display_quad_feature(frames_idx: List[int], locations: Tuple[TrackInstance, TrackInstance], loc_ids: int) -> None:
    """
    This function display a random track of length > 9.
    :param database:
    """
    for i, frame in enumerate(frames_idx):
        track_instance = locations[i]
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
        # cv2.imshow(f"{frame}", res)
        cv2.imwrite(f"display_quad_feature/Output_Image_{frame}_{loc_ids}.png", res)
    # cv2.waitKey(0)


def perform_bundle(database: DataBase):
    """
    This function perform bundle adjustment optimization on database.
    """
    stereo_k = get_stereo_k()
    current_transformation = np.hstack((np.eye(3), np.zeros((3, 1))))
    locations = np.zeros((3450, 3))
    jump = 19
    for i in range(0, 3450, jump):
        print(i)
        bundle_frames = list(range(i, min(i + jump, 3449) + 1))
        error_before, error_after, last_frame_pose = perform_bundle_window(database, stereo_k, bundle_frames)[:3]
        R = last_frame_pose.rotation().matrix()
        t = last_frame_pose.translation()
        R_t = np.hstack((R, t[:, None]))
        current_transformation = compute_extrinsic_matrix(R_t, current_transformation)
        locations[min(i + jump, 3449)] = current_transformation[:, 3]
    return locations.T


def find_end_keyframe(database: DataBase, frame_id: int):
    from collections import Counter
    max_frame_counter = Counter()
    for track_id in database.frames[frame_id].track_ids:
        curr_max_frame_id = database.tracks[track_id].frame_ids[-1]
        # if curr_max_frame_id-frame_id >= 19:
        #     return frame_id+19
        max_frame_counter[min(curr_max_frame_id, frame_id + 19)] += 1
    threshold = 20
    max_frame_set = {key for key, value in max_frame_counter.items() if value >= threshold}
    return max(max_frame_set)


def plot_local_error(real_locs, est_locs, title):
    jump = 19
    res = []
    dist_error = (real_locs - est_locs) ** 2
    error = np.sqrt(dist_error[0] + dist_error[1] + dist_error[2])
    for i in range(0, 3450, jump):
        res.append(error[min(i + jump, 3449)])
    x = np.arange(len(res)) * jump
    plt.plot(x, res)
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.clf()
