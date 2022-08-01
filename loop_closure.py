from typing import Iterable
from pose_graph import *
from gtsam import gtsam
from gtsam.utils import plot

MAHALANOBIS_DISTANCE_TEST = 500000
CONSENSUS_MATCHING_THRESHOLD = 0.6


def get_absolute_loop_closure_error(result: gtsam.Values, num_of_cameras: int = 3450, jump: int=JUMP) -> None:  # estimated_ext_mat is all poses (from create_pose_graph)
    """
    This function plot the absolute loop closure estimation error in X, Y, Z axis, the total error norm and the angle error.
    """
    estimated_locations = []
    estimated_ext_mat = []
    needed_indices = [i for i in range(0, num_of_cameras, jump)] + [num_of_cameras-1]

    for i in needed_indices:
        estimated_pose = result.atPose3(gtsam.symbol('x', i))
        estimated_locations.append(estimated_pose.translation())
        estimated_ext_mat.append(estimated_pose)
    real_ext_mat = np.array(read_ground_truth_extrinsic_mat())[needed_indices]
    real_locs = read_poses()[needed_indices]
    estimated_locations = np.array(estimated_locations)

    absolute_estimation_error(real_locs.T, estimated_locations.T, real_ext_mat, estimated_ext_mat, jump=JUMP, estimation_type="loop_closure")


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


def get_relative_covariance(c_n: Node, c_i: Node) -> Tuple[gtsam.noiseModel.Gaussian.Covariance, bool]:
    """
    This function return the relative covariance between poses c_i and c_n (the relative covariance is the sum of the
    covariances along the shortest path from c_i to c_n
    """
    return search(c_i, c_n)


def mahalanobis_distance(covariance: gtsam.noiseModel.Gaussian.Covariance, relative_pose: gtsam.Pose3) -> float:
    """
    This function calculate mahalanobis distance.
    """
    location = relative_pose.translation()
    angles = relative_pose.rotation().ypr()
    relative_vec = np.hstack((angles, location))
    return relative_vec.T @ covariance.information() @ relative_vec


def consensus_matching(img_idx_1: int, img_idx_2: int) -> Tuple[float, List[Tuple[TrackInstance, TrackInstance]]]:
    """
    This function perform consensus matching between img_idx_1 and img_idx_2.
    Return the percentage of inliers matches between the images, and their locations.
    """
    k = read_cameras()[0]
    quad, max_num_inliers = ransac(img_idx_1, img_idx_2, k, None)
    locs = []
    left_1_kp = []
    left_2_kp = []
    for idx in quad.stereo_pair2.left_image._get_pnp_inliers_kps_idx():
        kp_idx = idx
        r_idx = quad.stereo_pair2.get_left_right_kps_idx_dict()[kp_idx]
        kp_r_2 = quad.stereo_pair2.right_image.kps[r_idx]
        kp_l_2 = quad.stereo_pair2.left_image.kps[kp_idx]
        left_2_kp.append(kp_l_2)

        kp_l_1_idx = quad.left_left_kps_idx_dict[kp_idx]
        kp_l_1 = quad.stereo_pair1.left_image.kps[kp_l_1_idx]
        r_idx = quad.stereo_pair1.get_left_right_kps_idx_dict()[kp_l_1_idx]
        kp_r_1 = quad.stereo_pair1.right_image.kps[r_idx]
        left_1_kp.append(kp_l_1)

        locs.append((TrackInstance(kp_l_2.pt[0], kp_r_2.pt[0], kp_l_2.pt[1]), TrackInstance(kp_l_1.pt[0], kp_r_1.pt[0], kp_l_1.pt[1])))
    if max_num_inliers / len(quad.get_left_left_kps_idx_dict()) >= CONSENSUS_MATCHING_THRESHOLD:
        present_consensus_matching(img_idx_1, img_idx_2, np.array(left_1_kp), np.array(left_2_kp),
                                   quad.stereo_pair1.left_image.get_inliers_kps(FilterMethod.RECTIFICATION),
                                   quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.RECTIFICATION))
    return max_num_inliers / len(quad.get_left_left_kps_idx_dict()), locs, quad.get_relative_trans(), max_num_inliers


def new_bundle_window(database: DataBase, stereo_k: gtsam.Cal3_S2Stereo, bundle_frames: List[int], frame_poses: List[Union[gtsam.Pose3, None]], inliers_locs=None) -> Tuple[float, float, List[FloatNDArray], gtsam.NonlinearFactorGraph, gtsam.Values, List[int]]:
    """
    This function perform bundle adjustment optimization on a given frames (bundle_frames) using given key points locations.
    """
    start_frame = bundle_frames[0]
    start_frame_trans = database.frames[start_frame].transformation_from_zero
    initialEstimate = gtsam.Values()
    graph = gtsam.NonlinearFactorGraph()
    x_start = gtsam.symbol('x', start_frame)
    stereo_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1]))
    graph.add(gtsam.PriorFactorPose3(x_start, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))))
    track_id_to_point = {}
    initialEstimate.insert(x_start, gtsam.Pose3())
    frame_symbols = []
    factors = []
    to_remove_landmarks = set()

    x_end = gtsam.symbol('x', bundle_frames[1])
    frame_symbols.append(x_end)

    new_R, new_t = get_camera_to_global(frame_poses[1])
    # new_R, new_t = frame_poses[1][:, :3],frame_poses[1][:, 3]

    frame_pose = gtsam.Pose3(gtsam.Rot3(new_R), new_t)

    end_camera = gtsam.StereoCamera(frame_pose, stereo_k)
    end_rel_pose = end_camera.pose()
    initialEstimate.insert(x_end, end_rel_pose)

    for frame_idx in bundle_frames[::-1]:
        frame_symbol = x_start if frame_idx == start_frame else x_end
        for loc_idx, loc_tup in enumerate(inliers_locs):  # iterate over all the inliers
            display_quad_feature(bundle_frames, loc_tup, loc_idx)
            location = loc_tup[0 if frame_idx == start_frame else 1]
            if frame_idx != start_frame:
                s = gtsam.symbol('l', loc_idx)  # feature point
                point_3d = end_camera.backproject(gtsam.StereoPoint2(*location))
                track_id_to_point[loc_idx] = (s, point_3d)
                landmark = s
                x = point_3d[0]
                z = point_3d[2]
                if landmark in to_remove_landmarks or abs(x) > 25 or z > 87 or z < 0:
                    to_remove_landmarks.add(landmark)
                    continue
                initialEstimate.insert(s, point_3d)
            landmark = gtsam.symbol('l', loc_idx)
            factor = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(*location), stereo_model, frame_symbol, track_id_to_point[loc_idx][0], stereo_k)
            if landmark in to_remove_landmarks:
                continue
            if factor.error(initialEstimate) > 1000000:
                initialEstimate.erase(track_id_to_point[loc_idx][0])
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
    print("new bundle total error before optimization: ", error_before)
    print("new bundle total error after optimization: ", error_after)
    return error_before, error_after, last_frame_pose, graph, result, frame_symbols


def small_bundle(c_i_pose: gtsam.Pose3, c_n_pose: gtsam.Pose3, bundle_frames: List[int], database: DataBase, stereo_k: gtsam.Cal3_S2Stereo,
                 inliers_locs=None) -> Tuple[gtsam.Pose3, gtsam.noiseModel.Gaussian.Covariance]:
    """
    This function perform bundle adjustment between only two given frames.
    :param c_i_pose: frame c_i pose
    :param c_n_pose: frame c_n pose
    :param bundle_frames: list of two frames indices [i,n]
    :param database: database
    :param stereo_k: calibration matrix
    :param inliers_locs: the inliers locations (pixels in frames c_i, c_n)
    :return: the relative pose and the relative marginal covariance matrix.
    """
    error_before, error_after, last_frame_pose, graph, result, frame_symbols = new_bundle_window(database, stereo_k, bundle_frames, [c_i_pose, c_n_pose],
                                                                                                 inliers_locs)
    ci, cn = frame_symbols[-1], frame_symbols[0]
    c_i_new_pose = result.atPose3(ci)
    c_n_new_pose = result.atPose3(cn)
    marginals = gtsam.Marginals(graph, result)  # 6.1.1
    keys = gtsam.KeyVector()
    keys.append(ci)
    keys.append(cn)
    relative_pose = c_i_new_pose.between(c_n_new_pose)
    relative_marginal_covariance_mat = marginals.jointMarginalCovariance(keys).fullMatrix()
    relative_marginal_covariance_mat = np.linalg.inv(relative_marginal_covariance_mat)
    relative_marginal_covariance_mat = relative_marginal_covariance_mat[6:, 6:]
    relative_marginal_covariance_mat = np.linalg.inv(relative_marginal_covariance_mat)

    relative_marginal_covariance_mat = gtsam.noiseModel.Gaussian.Covariance(relative_marginal_covariance_mat, False)
    return relative_pose, relative_marginal_covariance_mat


def detect_loop_closure_candidates(all_poses: List[gtsam.Pose3], all_nodes: List[Node], pose_graph: gtsam.NonlinearFactorGraph, database: DataBase,
                                   stereo_k: gtsam.Cal3_S2Stereo, pose_result: gtsam.Values, rel_poses) -> Tuple[gtsam.Values, gtsam.NonlinearFactorGraph]:
    """
    This function search for a loop closure candidates.
    If it find two frame that passed mahalanobis distance test and consensus matching test it
    adds new edge to the pose graph, and perform optimization.
    """
    count = 0
    count_loop_closure_success = 0
    curr_result = pose_result
    loop_idx = []
    for c_n_idx in range(1, len(all_nodes)):
        for c_i_idx in range(c_n_idx):
            if c_n_idx - c_i_idx == 1:
                continue
            cov, success = get_relative_covariance(all_nodes[c_n_idx], all_nodes[c_i_idx])
            rel_pos = all_poses[c_i_idx].between(all_poses[c_n_idx])
            mahalanobis_dist = mahalanobis_distance(cov, rel_pos)
            if mahalanobis_dist < MAHALANOBIS_DISTANCE_TEST:
                print(f"Frames {c_n_idx * JUMP} and {c_i_idx * JUMP} are a {mahalanobis_dist} distance")
                inliers_percentage, inliers_locs, rel_pose_between_match, num_inliers = consensus_matching(min(c_n_idx * JUMP, 3449), min(c_i_idx * JUMP, 3449))
                if inliers_percentage >= CONSENSUS_MATCHING_THRESHOLD and num_inliers > 80:
                    print(f"Frames {c_n_idx * JUMP} and {c_i_idx * JUMP} are a possible match!")
                    print(f"matches number: {num_inliers}")
                    print(f"percentage number: {inliers_percentage}")
                    count_loop_closure_success += 1
                    relative_pose, covariance = small_bundle(rel_poses[c_i_idx], rel_pose_between_match, [min(c_i_idx * JUMP, 3449), min(c_n_idx * JUMP, 3449)],
                                                             database, stereo_k, inliers_locs)

                    all_nodes[c_i_idx].add_neighbor(all_nodes[c_n_idx], covariance)
                    factor = gtsam.BetweenFactorPose3(gtsam.symbol('x', min(c_i_idx * JUMP, 3449)), gtsam.symbol('x', min(c_n_idx * JUMP, 3449)), relative_pose,
                                                      covariance)
                    pose_graph.add(factor)
                    optimizer = gtsam.LevenbergMarquardtOptimizer(pose_graph, curr_result)
                    curr_result = optimizer.optimize()

                    plot_trajectory_from_result(curr_result,f"new_traj_results/after_frames_{c_n_idx * JUMP}_{c_i_idx * JUMP}_traj_2d")
                    plot_trajectory(1, curr_result, scale=2, title="Locations as a 3D")
                    plt.savefig(f"new_traj_results/after_frames_{c_n_idx * JUMP}_{c_i_idx * JUMP}_traj_3d.png")
                    plt.clf()


                    marginals = gtsam.Marginals(pose_graph, curr_result)
                    plot_trajectory(1, curr_result, marginals=marginals, scale=2, title="Locations as a 3D include the Covariance of the locations")
                    plt.savefig(f"new_traj_results/after_frames_{c_n_idx * JUMP}_{c_i_idx * JUMP}_traj_3d_cov.png")
                    plt.clf()
                    loop_idx.append(min(c_i_idx * JUMP, 3449))
                    loop_idx.append(min(c_n_idx * JUMP, 3449))
                count += 1
                # print("Mahalanobis distance:", mahalanobis_dist)
    print(count)
    print(f"{count_loop_closure_success} successful loop closure detected")
    plot_trajectory_from_result(curr_result, "loop_closure_traj", loop_idx=loop_idx)
    # plot_trajectory(1, result, scale=2, title="Locations as a 3D")
    # plt.savefig(f"after_frames_{c_n_idx * JUMP}_{c_i_idx * JUMP}_traj_3d.png")
    # plt.clf()
    return curr_result, pose_graph


def get_uncertainty_size(ci: gtsam.symbol, cn: gtsam.symbol, marginals: gtsam.Marginals) -> float:
    """
    This function return the uncertainty size between two frames.
    :param ci: frame symbol
    :param cn: frame symbol
    :param marginals:
    :return: the determinant of the relative covariance.
    """
    keys = gtsam.KeyVector()
    keys.append(ci)
    keys.append(cn)

    relative_marginal_covariance_mat = marginals.jointMarginalCovariance(keys).fullMatrix()
    relative_marginal_covariance_mat = np.linalg.inv(relative_marginal_covariance_mat)
    relative_marginal_covariance_mat = relative_marginal_covariance_mat[6:, 6:]
    relative_marginal_covariance_mat = np.linalg.inv(relative_marginal_covariance_mat)

    angles_marginal = relative_marginal_covariance_mat[:3, :3]
    locations_marginal = relative_marginal_covariance_mat[3:, 3:]
    return np.linalg.det(angles_marginal), np.linalg.det(locations_marginal)


def plot_uncertainty_graph(marginals_before: gtsam.Marginals, estimation_type:str) -> None:
    """
    This function plot the uncertainty graph.
    :param marginals_before: the covariance before optimization.
    :param marginals_after: the covariance after optimization.
    """
    uncer_locs, uncer_angles = [], []
    c0 = gtsam.symbol("x", 0)
    jump = JUMP
    for i in range(0, 3450, jump):
        cn = gtsam.symbol("x", min(i+jump, 3449))
        angle, loc = get_uncertainty_size(c0, cn, marginals_before)
        uncer_locs.append(loc)
        uncer_angles.append(angle)
    x = np.arange(len(uncer_locs)) * jump
    plt.figure()
    plt.ylim([0, 0.0011])
    plt.plot(x, uncer_locs)
    plt.xlabel('frame')
    plt.ylabel('uncertainty')
    plt.title(f"{estimation_type} - uncertainty size of locations vs frames")
    plt.savefig(f"uncer_loc_{estimation_type}.png")
    plt.clf()

    plt.ylim([0, 1.5*1e-17])
    plt.plot(x, uncer_angles)
    plt.xlabel('frame')
    plt.ylabel('uncertainty')
    plt.title(f"{estimation_type} - uncertainty size of angles vs frames")
    plt.savefig(f"uncer_angles_{estimation_type}.png")
    plt.clf()


def plot_trajectory_from_result(result: gtsam.Values, title: str, num_of_cameras: int = 3450, loop_idx: List[int]=None) -> None:
    """
    This function plot the trajectory from a given result.
    """
    locations = np.zeros((3450, 3))
    jump = JUMP
    for i in range(0, 3450, jump):
        locations[min(i + jump, 3449)] = result.atPose3(gtsam.symbol('x', i)).translation()

    l = read_poses()
    needed_indices = [i for i in range(0, num_of_cameras - 1, JUMP)] + [num_of_cameras - 1]
    real_poses = l[needed_indices].T
    loop_locs = locations[loop_idx]
    l2 = locations.T
    plot_trajectury(l2[0], l2[2], real_poses[0], real_poses[2], title, loop_locs=loop_locs)

def get_locations_from_result_graph(result: gtsam.Values) -> FloatNDArray:
    locations = np.zeros((3450, 3))
    jump = JUMP
    for i in range(0, 3450, jump):
        locations[min(i + jump, 3449)] = result.atPose3(gtsam.symbol('x', min(i + jump, 3449))).translation()
    return locations.T


def plot_all_results(pose_graph_result: gtsam.Values, loop_closure_results:gtsam.Values, initial_locations:FloatNDArray) -> None:
    num_of_cameras = 3450
    real_locations = read_poses()
    needed_indices = [i for i in range(0, num_of_cameras - 1, JUMP)] + [num_of_cameras - 1]

    real_poses = real_locations[needed_indices].T
    init_locs_key_frames = initial_locations[needed_indices].T
    pose_graph_locations = get_locations_from_result_graph(pose_graph_result)
    loop_closure_locations = get_locations_from_result_graph(loop_closure_results)

    plt.scatter(real_poses[0], real_poses[2], c='red', s=2, label='ground truth')
    plt.scatter(init_locs_key_frames[0], init_locs_key_frames[2], c='blue', s=2, label='initial estimate')
    plt.scatter(pose_graph_locations[0],  pose_graph_locations[2], c='orange', s=2, label='bundle adjustment')
    plt.scatter(loop_closure_locations[0], loop_closure_locations[2], c='green', s=2, label='loop closure')
    # plt.xlabel("z")
    # plt.ylabel("y")
    plt.title("trajectory of all estimations compared to the ground truth")
    plt.ylim([-100, 500])
    plt.legend()
    plt.savefig(f"traj_all.png")
    plt.clf()

def present_consensus_matching(img1_idx: int, img2_idx: int, inliers_img1: NDArray[cv2.KeyPoint], inliers_img2:  NDArray[cv2.KeyPoint], all_keypoints1:  NDArray[cv2.KeyPoint], all_keypoints2:  NDArray[cv2.KeyPoint]) -> None:
    """
    This function display the inliers and outliers keypoints.
    """
    img1 = read_images(img1_idx, ImageColor.RGB)[0]
    img2 = read_images(img2_idx, ImageColor.RGB)[0]
    cv2.drawKeypoints(img1, all_keypoints1, img1, (255, 255, 0))
    cv2.drawKeypoints(img1, inliers_img1, img1, (0, 128, 255))

    cv2.drawKeypoints(img2, all_keypoints2, img2, (255, 255, 0))  # out cyan
    cv2.drawKeypoints(img2, inliers_img2, img2, (0, 128, 255))
    cv2.imwrite(f"{img1_idx}_.png", img1)
    cv2.imwrite(f"{img2_idx}_.png", img2)


def plot_trajectory(fignum: int, values: gtsam.Values, scale: float = 1, marginals: gtsam.Marginals = None,
                    title: str = "Plot Trajectory", axis_labels: Iterable[str] = ("X axis", "Y axis", "Z axis")) -> None:
    """
    Plot a complete 2D/3D trajectory using poses in `values`.

    Args:
        fignum: Integer representing the figure number to use for plotting.
        values: Values containing some Pose2 and/or Pose3 values.
        scale: Value to scale the poses by.
        marginals: Marginalized probability values of the estimation.
            Used to plot uncertainty bounds.
        title: The title of the plot.
        axis_labels (iterable[string]): List of axis labels to set.
    """
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    axes.set_zlabel(axis_labels[2])
    axes.view_init(azim=270, elev=0)

    # Plot 2D poses, if any
    poses = gtsam.utilities.allPose2s(values)
    for key in poses.keys():
        pose = poses.atPose2(key)
        if marginals:
            covariance = marginals.marginalCovariance(key)
        else:
            covariance = None

        plot.plot_pose2_on_axes(axes,
                           pose,
                           covariance=covariance,
                           axis_length=scale)

    # Then 3D poses, if any
    poses = gtsam.utilities.allPose3s(values)
    for key in poses.keys():
        pose = poses.atPose3(key)
        if marginals:
            covariance = marginals.marginalCovariance(key)
        else:
            covariance = None

        plot.plot_pose3_on_axes(axes, pose, P=covariance, axis_length=scale)

    fig.suptitle(title)
    fig.canvas.set_window_title(title.lower())


if __name__ == '__main__':
    num_of_cameras = 3450
    database = open_database("database_neww")
    # database = create_database()
    # real_ext_mat = read_ground_truth_extrinsic_mat()
    # pnp_ext_mat = get_pnp_extrinsic_mat(database)
    # l = read_poses()
    # needed_indices = [i for i in range(0, num_of_cameras, JUMP)] + [num_of_cameras - 1]
    # real_poses = l[needed_indices].T
    # l3 = get_pnp_locations(database)
    # absolute_estimation_error(l.T, l3, real_ext_mat, pnp_ext_mat)
    # l3 = l3.T[[i for i in range(0, num_of_cameras, JUMP)]]
    # l3 = l3.T
    # plot_trajectury(l3[0], l3[2], real_poses[0], real_poses[2], "new_after_pnp")

    stereo_k = get_stereo_k()
    all_poses, all_nodes, graph, optimizer, rel_poses, l2, result = create_pose_graph(database, stereo_k)
    plot_trajectory_from_result(result, "new_after_pose")
    # get_absolute_pose_graph_error(all_poses)
    res, new_graph = detect_loop_closure_candidates(all_poses, all_nodes, graph, database, stereo_k, result, rel_poses)
    plot_trajectory_from_result(res, "new_after_loop")
    # get_absolute_loop_closure_error(res)
    #get_absolute_loop_closure_error(res, jump=JUMP)
    # marginals_before = gtsam.Marginals(graph, result)
    # plot_local_error(l, l2, "absolute error in meters before loop closure")

    #
    # locations = np.zeros((3450, 3))
    # for i in range(0, 3450, JUMP):
    #     locations[i] = transform_rt_to_location(database.frames[i].get_transformation_from_zero_bundle())
    #
    # l = read_poses().T
    # l2 = locations.T
    # plot_trajectury(l2[0], l2[2], l[0], l[2])  # ploting the trajectory after bundle optimization compared to ground truth