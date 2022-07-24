import numpy as np

# import pose_graph
from database import *
from gtsam import gtsam, utils
from gtsam.gtsam import NonlinearFactorGraph, GenericStereoFactor3D
# from gtsam.gtsam import BetweenFactorPose3
from gtsam.noiseModel import Gaussian
from typing import List


from collections import defaultdict
def default_dict_struct(): return defaultdict(list_struct)
def list_struct(): return [int(), int()]


def get_stereo_k() -> gtsam.Cal3_S2Stereo:
    """
    This function create and return calibration stereo camera.
    """
    k, m1, m2 = read_cameras()
    f_x, f_y, skew, c_x, c_y, baseline = k[0][0], k[1][1], k[0][1], k[0][2], k[1][2], m2[0][3]
    stereo_k = gtsam.Cal3_S2Stereo(f_x, f_y, skew, c_x, c_y, -baseline)
    return stereo_k


def get_pnp_locations(database:DataBase) -> FloatNDArray:
    """
    This function returns all location after PnP estimation from the database.
    """
    locations = np.zeros((3450, 3))
    for i in range(3450):
        locations[i] = transform_rt_to_location(database.frames[i].get_transformation_from_zero())
    return locations.T


def get_pnp_extrinsic_mat(database:DataBase) -> List[gtsam.Pose3]:
    """
    This function create and returns all extrinsic matrices after PnP estimation from the database.
    """
    extrinsic_matrix_arr = []
    for i in range(3450):
        new_R, new_t = get_camera_to_global(database.frames[i].get_transformation_from_zero())
        i_to_zero_trans = np.hstack((new_R, new_t))
        new_R = i_to_zero_trans[:, :3]
        new_t = i_to_zero_trans[:, 3]
        frame_pose = gtsam.Pose3(gtsam.Rot3(new_R), new_t)
        extrinsic_matrix_arr.append(frame_pose)
    return extrinsic_matrix_arr


def read_ground_truth_extrinsic_mat(first_index: int = 0, last_index: int = 3450) -> List[gtsam.Pose3]:
    """
    This function create and returns all extrinsic matrices of ground truth.
    """
    extrinsic_matrix_arr = []
    i = 0
    with open(os.path.join(POSES_PATH, '00.txt')) as f:
        for l in f.readlines()[first_index:last_index]:
            l = l.split()
            extrinsic_matrix = np.array([float(i) for i in l])
            extrinsic_matrix = extrinsic_matrix.reshape(3, 4)
            new_R, new_t = get_camera_to_global(extrinsic_matrix)
            i_to_zero_trans = np.hstack((new_R, new_t))
            new_R = i_to_zero_trans[:, :3]
            new_t = i_to_zero_trans[:, 3]
            frame_pose = gtsam.Pose3(gtsam.Rot3(new_R), new_t)
            extrinsic_matrix_arr.append(frame_pose)
            i += 1
    return extrinsic_matrix_arr


def absolute_estimation_error(real_locations: FloatNDArray, estimated_locations: FloatNDArray, real_ext_mat: List[gtsam.Pose3], estimated_ext_mat: List[gtsam.Pose3], num_of_cameras: int = 3450, jump:int = 1, estimation_type:str="PnP") -> None:
    """
    This function plot the absolute estimation error in X, Y, Z axis, the total error norm and the angle error.
    """
    sq_dist_error = (real_locations - estimated_locations) ** 2
    norm = np.sqrt(sq_dist_error[0] + sq_dist_error[1] + sq_dist_error[2])
    x_error = np.sqrt(sq_dist_error[0])
    y_error = np.sqrt(sq_dist_error[1])
    z_error = np.sqrt(sq_dist_error[2])
    angle_error = np.zeros((num_of_cameras, 3))

    # angle error calc
    j=0
    for i in range(0, num_of_cameras, jump):
        estimate_angles = estimated_ext_mat[j].rotation().ypr()
        real_angles = real_ext_mat[i].rotation().ypr()
        angle_error[i] = real_angles-estimate_angles
        j+=1
    angle_error = (angle_error.T) ** 2
    angle_error = np.sqrt(angle_error[0] + angle_error[1] + angle_error[2])

    plt.figure(figsize=(15, 5))
    plt.ylim([0, 36])
    plt.plot(norm, label="norm")
    plt.plot(x_error, label="x error")
    plt.plot(y_error, label="y error")
    plt.plot(z_error, label="z error")
    plt.plot(angle_error, label="angle error")
    plt.legend()
    plt.title(f"Absolute {estimation_type} estimation error")
    plt.savefig(f"absolute_{estimation_type}_estimation_error.png")
    plt.clf()


def calc_error(location: TrackInstance, camera: gtsam.Cal3_S2Stereo, point_3d) -> float:
    """
    This function project a given point and return the error norm.
    """
    projected_p = camera.project(point_3d)
    left_pt = np.array([projected_p.uL(), projected_p.v()])
    left_location = np.array([location.x_l, location.y])
    return calculate_norm(left_pt, left_location)


def projection_error_pnp_vs_bundle(start_frame_idx: int, end_frame_idx: int, database: DataBase) -> None:
    """
    This function plot the mean projection error of the different track links as a function of distance from the
    reference frame, for all tracks in frame: start_frame_idx.
    for PnP and bundle estimation.
    """
    pnp_left_error, bundle_left_error = default_dict_struct(), default_dict_struct()  # track_len: [error, track_count]
    stereo_k = get_stereo_k()
    for track_id in database.frames[start_frame_idx].track_ids:
        track = database.tracks[track_id]
        end_frame_trans_pnp = database.frames[track.frame_ids[-1]].transformation_from_zero
        end_frame_trans_bundle = database.frames[track.frame_ids[-1]].get_transformation_from_zero_bundle()
        pnp_start_frame_stereo_camera = create_stereo_camera(database, track.frame_ids[0], stereo_k, end_frame_trans_pnp)[0]
        bundle_start_frame_stereo_camera = create_stereo_camera(database, track.frame_ids[0], stereo_k, end_frame_trans_bundle)[0]
        point_3d_pnp = pnp_start_frame_stereo_camera.backproject(gtsam.StereoPoint2(*track.track_instances[0]))
        point_3d_bundle = bundle_start_frame_stereo_camera.backproject(gtsam.StereoPoint2(*track.track_instances[0]))

        for i, frame_idx in enumerate(track.frame_ids):
            pnp_camera = create_stereo_camera(database, frame_idx, stereo_k, end_frame_trans_pnp)[0]
            bundle_camera = create_stereo_camera(database, frame_idx, stereo_k, end_frame_trans_bundle, True)[0]
            location = track.track_instances[i]
            pnp_left_error[i] = [pnp_left_error[i][0]+calc_error(location, pnp_camera, point_3d_pnp), pnp_left_error[i][1]+1]
            bundle_left_error[i] += [bundle_left_error[i][0]+calc_error(location, bundle_camera, point_3d_bundle), bundle_left_error[i][1]+1]

    bundle_error, pnp_error = [], []
    for k, v in sorted(pnp_left_error.items()):
        pnp_error.append(v[0]/v[1])
    for k, v in sorted(bundle_left_error.items()):
        bundle_error.append(v[0]/v[1])

    plt.plot(bundle_error)
    print(pnp_left_error)
    plt.plot(pnp_error)
    plt.xlabel('distance from ref')
    plt.ylabel('mean projection error')
    plt.title("projection error vs dist")
    plt.savefig("projection_vs_dist.png")
    plt.clf()


def relative_estimation_error(sequence_len: int, real_ext_mat: List[gtsam.Pose3], estimated_ext_mat: List[gtsam.Pose3], estimation_type: str) -> None:
    # real_relative_poses = []
    # estimated_relative_poses = []
    angle_error = np.zeros((3450-sequence_len, 3))

    real_locations = np.zeros((3450-sequence_len, 3))
    estimated_locations = np.zeros((3450-sequence_len, 3))
    for i in range(0, 3450-sequence_len):
        real_pose_c0 = real_ext_mat[i]
        real_pose_ck = real_ext_mat[i+sequence_len]
        real_relative_pose = real_pose_c0.between(real_pose_ck)
        # real_relative_poses.append(real_relative_pose)
        real_locations[i] = real_relative_pose.translation()
        estimate_angles = real_relative_pose.rotation().ypr()

        est_pose_c0 = estimated_ext_mat[i]
        est_pose_ck = estimated_ext_mat[i+sequence_len]
        est_relative_pose = est_pose_c0.between(est_pose_ck)
        # estimated_relative_poses.append(est_relative_pose)
        estimated_locations[i] = est_relative_pose.translation()
        real_angles = est_relative_pose.rotation().ypr()

        # angle_error[i] = (real_angles-estimate_angles) /  # todo

    real_locations = real_locations.T
    estimated_locations = estimated_locations.T
    error = np.abs(real_locations - estimated_locations)*100 / real_locations  # todo
    print(error[1])
    sq_error = error**2
    norm = np.sqrt(sq_error[0] + sq_error[1] + sq_error[2])
    x_error = error[0]
    y_error = error[1]
    z_error = error[2]
    # angle_error = (angle_error.T) ** 2
    # angle_error = np.sqrt(angle_error[0] + angle_error[1] + angle_error[2])

    plt.figure(figsize=(15, 5))
    plt.plot(norm, label="norm")
    plt.plot(x_error, label="x error")
    plt.plot(y_error, label="y error")
    plt.plot(z_error, label="z error")
    # plt.plot(angle_error, label="angle error")
    plt.legend()
    plt.title(f"relative {estimation_type} estimation error for sequence length of {sequence_len}")
    plt.savefig(f"relative_{estimation_type}_estimation_error_{sequence_len}.png")
    plt.clf()


def reprojection_error(database: DataBase) -> None:
    """
    This function calculate the reprojection error of track in database.
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
    plt.clf()


def find_end_keyframe(database: DataBase, frame_id: int) -> int:
    """
    This function finds the end keyframe given a start keyframe index
    :param database: database
    :param frame_id: start keyframe index
    :return: end keyframe index
    """
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


def get_camera_to_global(R_t: FloatNDArray) -> Tuple[FloatNDArray, FloatNDArray]:
    """
    This function convert given transformation: global->camera to camera->global.
    """
    R = R_t[:, :3]
    t = R_t[:, 3]
    new_R = R.T
    new_t = - new_R @ t
    return new_R, new_t[:, None]


def create_stereo_camera(database: DataBase, frame_idx: int, stereo_k: gtsam.Cal3_S2Stereo, start_frame_trans: FloatNDArray, after_bundle: bool=False) -> Tuple[gtsam.StereoCamera, gtsam.Pose3]:
    """
    This function create and return stereo camera for a given frame.
    """
    curr_frame = database.frames[frame_idx]
    new_R, new_t = get_camera_to_global(curr_frame.transformation_from_zero)
    if after_bundle:
        new_R, new_t = get_camera_to_global(curr_frame.transformation_from_after_bundle)
    i_to_zero_trans = np.hstack((new_R, new_t))
    i_to_start_trans = compute_extrinsic_matrix(i_to_zero_trans, start_frame_trans)
    new_R = i_to_start_trans[:, :3]
    new_t = i_to_start_trans[:, 3]

    frame_pose = gtsam.Pose3(gtsam.Rot3(new_R), new_t)
    return gtsam.StereoCamera(frame_pose, stereo_k), frame_pose


def perform_bundle_window(database: DataBase, stereo_k: gtsam.Cal3_S2Stereo, bundle_frames: List[int], current_transformation: FloatNDArray=np.hstack((np.eye(3), np.zeros((3, 1))))) -> Tuple[
    float, float, List[FloatNDArray], gtsam.NonlinearFactorGraph, gtsam.Values, List[int], FloatNDArray]:
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
    # todo need to save R_t after bundle to database
    # print(current_transformation)
    # for frame_symbol, frame_idx in zip(frame_symbols[::-1][:-1], bundle_frames[:-1]):
    #
    #     np.set_printoptions(precision=2)
    #     print(frame_idx)
    #     print(current_transformation)
    #     pose_graph.update_database_pose(database, current_transformation, frame_idx)
    #     pose = result.atPose3(frame_symbol)
    #     R = pose.rotation().matrix()
    #     t = pose.translation()
    #     R_t = np.hstack((R, t[:, None])) # i->start
    #     current_transformation = compute_extrinsic_matrix(R_t, current_transformation)

    return error_before, error_after, last_frame_pose, graph, result, frame_symbols, current_transformation


def perform_bundle(database: DataBase) -> FloatNDArray:
    """
    This function perform bundle adjustment optimization on database.
    """
    stereo_k = get_stereo_k()
    locations = np.zeros((3450, 3))
    jump = 19
    current_transformation = np.hstack((np.eye(3), np.zeros((3, 1))))
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


def plot_local_error(real_locs: FloatNDArray, est_locs: FloatNDArray, title: str = "local_error") -> None:
    """
    This function plot the local error (the norm between the ground truth and the estimate).
    :param real_locs: ground truth locations.
    :param est_locs: estimated locations.
    :param title: wanted plot title
    """
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



if __name__ == '__main__':
    # with open("db_after_bundle.db", "rb") as file:
    #     database = pickle.load(file)

    database = open_database()
    # reprojection_error(database)
    # plot_local_error(l, l3, "check")
    # 1 graph
    # projection_error_pnp_vs_bundle(0,28, database)

    print("a")
    l = read_poses().T
    # l2 = perform_bundle(database)
    l3 = get_pnp_locations(database)
    real_ext_mat = read_ground_truth_extrinsic_mat()
    # 2 graph
    pnp_ext_mat = get_pnp_extrinsic_mat(database)
    plot_trajectury(l3[0], l3[2], l[0], l[2])  # ploting the trajectory after bundle optimization compared to ground truth
    # absolute_estimation_error(l, l3, real_ext_mat, pnp_ext_mat)  # ploting the distance error in meters
    relative_estimation_error(800, real_ext_mat, pnp_ext_mat, "PnP")