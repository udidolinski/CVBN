import random
import numpy as np
from database import *
from gtsam import gtsam, utils
from gtsam.noiseModel import Gaussian
from collections import defaultdict
from typing import List
JUMP = 19



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
    for i in range(0, 3450):
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


def get_bundle_extrinsic_mat(database:DataBase) -> List[gtsam.Pose3]:
    """
    This function create and returns all extrinsic matrices after PnP estimation from the database.
    """
    extrinsic_matrix_arr = []
    for i in range(3450):
        new_R, new_t = get_camera_to_global(database.frames[i].get_transformation_from_zero_bundle())
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

            temp_R = gtsam.Rot3(extrinsic_matrix[:, :3])
            # if i % 19 == 0:
            #     print(f"real angles before trans: {temp_R.ypr()}")

            new_R, new_t = get_camera_to_global(extrinsic_matrix)
            i_to_zero_trans = np.hstack((new_R, new_t))
            new_R = i_to_zero_trans[:, :3]
            new_t = i_to_zero_trans[:, 3]
            frame_pose = gtsam.Pose3(gtsam.Rot3(new_R), new_t)
            # if i % 19 == 0:
            #     print(f"real angles before trans: {gtsam.Rot3(new_R).ypr()}")
            extrinsic_matrix_arr.append(frame_pose)
            i += 1
    return extrinsic_matrix_arr


def absolute_estimation_error(real_locations: FloatNDArray, estimated_locations: FloatNDArray, real_ext_mat: List[gtsam.Pose3], estimated_ext_mat: List[gtsam.Pose3], estimation_type:str, num_of_cameras: int = 3450, jump:int = 1) -> None:
    """
    This function plot the absolute estimation error in X, Y, Z axis, the total error norm and the angle error.
    """
    sq_dist_error = (real_locations - estimated_locations) ** 2
    norm = np.sqrt(sq_dist_error[0] + sq_dist_error[1] + sq_dist_error[2])
    x_error = np.sqrt(sq_dist_error[0])
    y_error = np.sqrt(sq_dist_error[1])
    z_error = np.sqrt(sq_dist_error[2])
    angle_error = []

    # angle error calc
    j=0
    for i in list(range(0, num_of_cameras-1, jump))+[num_of_cameras-1]:
        # estimate_angles = estimated_ext_mat[j].rotation().ypr()
        # real_angles = real_ext_mat[j].rotation().ypr()
        # angle_error.append(np.abs(real_angles)-np.abs(estimate_angles))
        # relative_pose = real_ext_mat[j].between(estimated_ext_mat[j])

        estimate_angles = estimated_ext_mat[j].rotation().ypr()
        real_angles = real_ext_mat[j].rotation().ypr()
        mid_angle = abs(real_angles - estimate_angles)

        # print(f"loop angles est index={i}: {estimate_angles}")
        # print(f"loop angles real index={i}: {real_angles}")
        angle_error.append(mid_angle)
        for k in range(3):
            if angle_error[j][k] > np.pi:
                angle_error[j][k] = 2*np.pi - angle_error[j][k]

        # print(f"loop angles relative index={i}: {angle_error[j]}")
        #
        # if real_angles[0] < -1 and estimate_angles[0] > 1:
        #     print(f"est angles: {estimate_angles}")
        #     print(f"real angles: {real_angles}")
        #
        #     print(f"new deg: { angle_error[j]}")

        j+=1

    angle_error = np.array(angle_error) * 180 / np.pi
    angle_y_error = angle_error.T[1] # it's already absolute values
    angle_error = (angle_error ** 2).T
    angle_error = np.sqrt(angle_error[0] + angle_error[1] + angle_error[2])
    x = [i for i in range(0, num_of_cameras, JUMP)] + [num_of_cameras-1]
    if estimation_type == "PnP" or estimation_type == "bundle_adjustment":
        x = [i for i in range(num_of_cameras)]

    plt.figure(figsize=(15, 5))
    plt.ylim([0, 60])
    plt.plot(x, norm, label="norm")
    plt.plot(x, x_error, label="x error")
    plt.plot(x, y_error, label="y error")
    plt.plot(x, z_error, label="z error")
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('Error (m)')
    plt.title(f"Absolute {estimation_type} estimation error")
    plt.savefig(f"absolute_{estimation_type}_estimation_error.png")
    plt.clf()

    plt.figure(figsize=(15, 5))
    plt.plot(x, angle_error, label="angle error")
    mean_angle_error = round(np.mean(angle_error), 2)
    median_angle_error = round(np.median(angle_error), 2)
    plt.plot(x, [mean_angle_error for _ in range(len(angle_error))], label=f'Mean={mean_angle_error}', linestyle='--')
    plt.plot(x, [median_angle_error for _ in range(len(angle_error))], label=f'Median={median_angle_error}', linestyle='--')
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('Error (deg)')
    plt.title(f"Absolute {estimation_type} angle estimation error")
    plt.savefig(f"absolute_{estimation_type}_angle_estimation_error.png")
    plt.clf()

    plt.figure(figsize=(15, 5))
    plt.plot(x, angle_y_error, label="angle y error")
    mean_angle_y_error = round(np.mean(angle_y_error), 2)
    median_angle_y_error = round(np.median(angle_y_error), 2)
    plt.plot(x, [mean_angle_y_error for _ in range(len(angle_y_error))], label=f'Mean={mean_angle_y_error}', linestyle='--')
    plt.plot(x, [median_angle_y_error for _ in range(len(angle_y_error))], label=f'Median={median_angle_y_error}', linestyle='--')
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('Error (deg)')
    plt.title(f"Absolute {estimation_type} angle y estimation error")
    plt.savefig(f"absolute_{estimation_type}_angle_y_estimation_error.png")
    plt.clf()


def new_reprojection_error(database: DataBase, is_estimation_type_bundle: bool):
    """
    This function calculate the reprojection and factor median error for estimation_type.
    :param database:
    :return:
    """
    tracks_bigger_than_40 = list(np.array(database.tracks)[np.array(database.tracks) > 40])
    random_tracks = np.array(tracks_bigger_than_40)[np.random.choice(len(tracks_bigger_than_40), 10, replace=False)]
    left_errors, right_errors = defaultdict(list),  defaultdict(list)  # track len : [errors]
    factor_errors = defaultdict(list)
    k, m1, m2 = read_cameras()
    f_x, f_y, skew, c_x, c_y, baseline = k[0][0], k[1][1], k[0][1], k[0][2], k[1][2], m2[0][3]
    stereo_k = gtsam.Cal3_S2Stereo(f_x, f_y, skew, c_x, c_y, -baseline)
    for random_track in random_tracks:
        start_frame_trans = database.frames[random_track.frame_ids[0]].transformation_from_zero if not is_estimation_type_bundle \
            else database.frames[random_track.frame_ids[0]].transformation_from_zero

        last_frame_stereo_camera, last_frame_pose = create_stereo_camera(database, random_track.frame_ids[-1], stereo_k,
                                                                         start_frame_trans, is_estimation_type_bundle)
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
            curr_camera, frame_pose = create_stereo_camera(database, frame_idx, stereo_k, start_frame_trans, is_estimation_type_bundle)
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

        for i, factor in enumerate(factors[::-1]):
            factor_errors[i].append(factor.error(initialEstimate))

        for i, (l_e, r_e) in enumerate(zip(left_error[::-1], right_error[::-1])):
            left_errors[i].append(l_e)
            right_errors[i].append(r_e)

    medians_left = np.zeros(len(left_errors.keys()))
    for k, v in left_errors.items():
        medians_left[k] = np.median(v)
    medians_right = np.zeros(len(right_errors.keys()))
    for k, v in right_errors.items():
        medians_right[k] = np.median(v)
    factor_medians = np.zeros(len(factor_errors.keys()))
    for k, v in factor_errors.items():
        factor_medians[k] = np.median(v)

    name = "_bundle_adjustment" if is_estimation_type_bundle  else "_PnP"
    plot_projection_error('reprojection error', medians_left, medians_right, file_name="reprojection_error"+name)
    plot_projection_error('factor error', factor_medians, file_name="factor_error"+name)


def relative_estimation_error(sequence_len: int, real_ext_mat: List[gtsam.Pose3], estimated_ext_mat: List[gtsam.Pose3], estimation_type: str) -> None:
    """
    This function calculate the relative pose estimation error compared to the ground truth relative pose
    evaluated on sequence length of sequence_len.
    """
    # real_relative_poses = []
    # estimated_relative_poses = []

    all_distance_travelled = []
    norm_all_total_distance_error = []
    x_all_total_distance_error = []
    y_all_total_distance_error = []
    z_all_total_distance_error = []
    all_angle_error = []
    all_y_angle_error = []
    for i in range(0, 3450-sequence_len):
        distance_travelled = 0
        norm_total_distance_error = 0
        x_total_distance_error = 0
        y_total_distance_error = 0
        z_total_distance_error = 0
        angle_error = 0
        angle_y_error = 0
        for j in range(sequence_len):
            real_pose_first = real_ext_mat[i + j]
            real_pose_second = real_ext_mat[i + j + 1]

            real_loc_first = real_pose_first.translation()
            real_loc_second = real_pose_second.translation()
            distance_travelled += np.linalg.norm(real_loc_second - real_loc_first)


            est_pose_first = estimated_ext_mat[i + j]
            est_pose_second = estimated_ext_mat[i + j + 1]

            relative_est = est_pose_first.between(est_pose_second)
            relative_real = real_pose_first.between(real_pose_second)

            relative_error_pose = relative_est.between(relative_real)



            norm_total_distance_error += np.linalg.norm(relative_error_pose.translation())
            x_total_distance_error += abs(relative_error_pose.translation()[0])
            y_total_distance_error += abs(relative_error_pose.translation()[1])
            z_total_distance_error += abs(relative_error_pose.translation()[2])
            curr_angles = relative_error_pose.rotation().ypr()
            angle_error += (np.linalg.norm(curr_angles) * 180) / np.pi
            angle_y_error += (abs(curr_angles[1]) * 180) / np.pi


        all_distance_travelled.append(distance_travelled)
        norm_all_total_distance_error.append(norm_total_distance_error)
        x_all_total_distance_error.append(x_total_distance_error)
        y_all_total_distance_error.append(y_total_distance_error)
        z_all_total_distance_error.append(z_total_distance_error)
        all_angle_error.append(angle_error)
        all_y_angle_error.append(angle_y_error)

        # angle_error[i] = (real_angles-estimate_angles) /  # todo

    # real_locations = real_locations.T
    # estimated_locations = estimated_locations.T
    # error = np.abs(real_locations - estimated_locations)*100 / real_locations
    # print(error[1])
    # sq_error = error**2
    # norm = np.sqrt(sq_error[0] + sq_error[1] + sq_error[2])
    # x_error = error[0]
    # y_error = error[1]
    # z_error = error[2]
    # angle_error = (angle_error.T) ** 2
    # angle_error = np.sqrt(angle_error[0] + angle_error[1] + angle_error[2])
    per_norm = 100 * np.array(norm_all_total_distance_error)/np.array(all_distance_travelled)
    per_x = 100 * np.array(x_all_total_distance_error)/np.array(all_distance_travelled)
    per_y = 100 * np.array(y_all_total_distance_error)/np.array(all_distance_travelled)
    per_z = 100 * np.array(z_all_total_distance_error)/np.array(all_distance_travelled)

    plt.figure(figsize=(15, 5))
    plt.plot(per_norm, label="distance error norm")
    plt.plot(per_x, label="x error")
    plt.plot(per_y, label="y error")
    plt.plot(per_z, label="z error")
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('Error (%)')
    plt.title(f"relative {estimation_type} estimation error for sequence length of {sequence_len}")
    plt.savefig(f"relative_{estimation_type}_estimation_error_{sequence_len}.png")
    plt.clf()


    per_deg = np.array(all_angle_error)/np.array(all_distance_travelled)

    average_total_deg = round(np.mean(per_deg), 4)
    median_total_deg = round(np.median(per_deg), 4)
    # print(f"The average error (degree) of sequence length {sequence_len} is: {average_total_deg}")
    plt.plot(per_deg, label="angle error")
    plt.plot([average_total_deg for _ in range(len(per_deg))], label=f'Mean={average_total_deg}', linestyle='--')
    plt.plot([median_total_deg for _ in range(len(per_deg))], label=f'Median={median_total_deg}', linestyle='--')
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('Error (deg/m)')
    plt.title(f"relative {estimation_type} estimation angle error for sequence length of {sequence_len}")
    plt.savefig(f"relative_{estimation_type}_estimation_angle_error_{sequence_len}.png")
    plt.clf()
    # print(f"The average error (norm) of sequence length {sequence_len} is: {average_total_norm}")

    per_y_deg = np.array(all_y_angle_error)/np.array(all_distance_travelled)

    average_total_y_deg = round(np.mean(per_y_deg), 5)
    median_total_y_deg = round(np.median(per_y_deg), 5)
    # print(f"The average error (degree) of sequence length {sequence_len} is: {average_total_deg}")
    plt.plot(per_y_deg, label="angle y error")
    plt.plot([average_total_y_deg for _ in range(len(per_y_deg))], label=f'Mean={average_total_y_deg}', linestyle='--')
    plt.plot([median_total_y_deg for _ in range(len(per_y_deg))], label=f'Median={median_total_y_deg}', linestyle='--')
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('Error (deg/m)')
    plt.title(f"relative {estimation_type} estimation angle error for sequence length of {sequence_len}")
    plt.savefig(f"relative_{estimation_type}_estimation_y_angle_error_{sequence_len}.png")
    plt.clf()

    average_total_norm = round(np.mean(per_norm), 3)
    median_total_norm = round(np.median(per_norm), 3)
    plt.figure(figsize=(15, 5))
    plt.plot(per_norm, label="distance error norm")
    plt.plot([average_total_norm for _ in range(len(per_norm))], label=f'Mean={average_total_norm}', linestyle='--')
    plt.plot([median_total_norm for _ in range(len(per_norm))], label=f'Median={median_total_norm}', linestyle='--')
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('Error (%)')
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
        # if curr_max_frame_id-frame_id >= JUMP:
        #     return frame_id+JUMP
        max_frame_counter[min(curr_max_frame_id, frame_id + JUMP)] += 1
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

def update_database_pose(database: DataBase, current_trans_to_zero: FloatNDArray, index: int) -> None:
    """
    This function update the transformation_from_zero of frame index.
    """
    new_R, new_t = get_camera_to_global(current_trans_to_zero)
    database.frames[index].set_transformation_from_zero_after_bundle(np.hstack((new_R, new_t)))


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
    stereo_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.5, 0.1]))
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
    new = current_transformation
    for frame_symbol, frame_idx in zip(frame_symbols[::-1], bundle_frames):
        # np.set_printoptions(precision=2)
        # print(frame_idx)
        # print(frame_symbol)
        # print(new)

        pose = result.atPose3(frame_symbol)
        R = pose.rotation().matrix()
        t = pose.translation()
        R_t = np.hstack((R, t[:, None]))  # i -> start
        new = compute_extrinsic_matrix(R_t, current_transformation)
        update_database_pose(database, new, frame_idx)


    return error_before, error_after, last_frame_pose, graph, result, frame_symbols, new


def perform_bundle(database: DataBase) -> FloatNDArray:
    """
    This function perform bundle adjustment optimization on database.
    """
    stereo_k = get_stereo_k()
    locations = np.zeros((3450, 3))
    jump = JUMP
    current_transformation = np.hstack((np.eye(3), np.zeros((3, 1))))
    current_transformation2 = np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(0, 3450, jump):
        print(i)
        bundle_frames = list(range(i, min(i + jump, 3449) + 1))
        error_before, error_after, last_frame_pose, graph, result, frame_symbols, current_transformation2 = perform_bundle_window(database, stereo_k, bundle_frames, current_transformation2)
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
    jump = JUMP
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

    database = open_database("SIFT_4_0.5_1_0.99")
    # reprojection_error(database)
    # plot_local_error(l, l3, "check")
    # 1 graph
    # projection_error_pnp_vs_bundle(0,28, database)

    # print("a")
    l = read_poses().T
    l3 = get_pnp_locations(database)
    plot_trajectury(l3[0], l3[2], l[0], l[2], "before_bundle_newwwww")
    l2 = perform_bundle(database)
    plot_trajectury(l2[0], l2[2], l[0], l[2], "after_bundle_newwwww")

    # real_ext_mat = read_ground_truth_extrinsic_mat()
    # 2 graph
    # pnp_ext_mat = get_pnp_extrinsic_mat(database)
    # plot_trajectury(l3[0], l3[2], l[0], l[2])  # ploting the trajectory after bundle optimization compared to ground truth
    # # absolute_estimation_error(l, l3, real_ext_mat, pnp_ext_mat)  # ploting the distance error in meters
    # relative_estimation_error(800, real_ext_mat, pnp_ext_mat, "PnP")