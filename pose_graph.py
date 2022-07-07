from bundle_adjustment import *
from gtsam import gtsam, utils

def extract_relative_pose(database: DataBase, stereo_k: gtsam.Cal3_S2Stereo, first, last):  # q 6.1
    """
    This function extract the relative pose and the relative marginal covariance matrix between the frames first and last.
    """
    bundle_frames = list(range(first, last + 1))
    graph, result, frame_symbols = perform_bundle_window(database, stereo_k, bundle_frames)[3:]
    marginals = gtsam.Marginals(graph, result)  # 6.1.1
    # plot_trajectory(1, result, marginals=marginals, scale=2, title="Locations as a 3D include the Covariance of the locations")  # 6.1.2
    # plt.show()
    c0, ck = frame_symbols[-1], frame_symbols[0]
    pose_c0 = result.atPose3(c0)
    pose_ck = result.atPose3(ck)
    keys = gtsam.KeyVector()
    keys.append(c0)
    keys.append(ck)
    relative_pose = pose_c0.between(pose_ck)
    relative_marginal_covariance_mat = marginals.jointMarginalCovariance(keys).fullMatrix()
    relative_marginal_covariance_mat = np.linalg.inv(relative_marginal_covariance_mat)
    relative_marginal_covariance_mat = relative_marginal_covariance_mat[6:, 6:]
    relative_marginal_covariance_mat = np.linalg.inv(relative_marginal_covariance_mat)
    relative_marginal_covariance_mat = gtsam.noiseModel.Gaussian.Covariance(relative_marginal_covariance_mat, False)
    np.set_printoptions(precision=5, suppress=True)
    # print("relative pose: \n", relative_pose)  # 6.1.3
    # print("the relative marginal covariance matrix: \n", relative_marginal_covariance_mat)  # 6.1.3
    return pose_ck, ck, relative_marginal_covariance_mat


def update_database_pose(database: DataBase, current_trans_to_zero: FloatNDArray, index: int) -> None:
    """
    This function update the transformation_from_zero of frame index.
    """
    new_R, new_t = get_camera_to_global(current_trans_to_zero)
    database.frames[index].set_transformation_from_zero(np.hstack((new_R, new_t)))


def create_pose_graph(database: DataBase, stereo_k: gtsam.StereoCamera) -> Tuple[
    List[gtsam.Pose3], List[Node], gtsam.NonlinearFactorGraph, gtsam.LevenbergMarquardtOptimizer, List[gtsam.Pose3], FloatNDArray]:
    """
    This function create a pose graph and perform bundle adjustment optimization of that graph.
    """
    initial_poses = np.zeros((3450, 3))
    current_transformation = np.hstack((np.eye(3), np.zeros((3, 1))))
    x_start = gtsam.symbol('x', 0)
    initialEstimate = gtsam.Values()
    graph = gtsam.NonlinearFactorGraph()

    graph.add(gtsam.PriorFactorPose3(x_start, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))))
    # graph.add(gtsam.PriorFactorPose3(x_start, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1, 1, 1, 1]))))
    initialEstimate.insert(x_start, gtsam.Pose3())
    curr_pose = gtsam.Pose3()
    curr_symbol = x_start
    start_node = Node(x_start, {})
    curr_node = start_node
    all_nodes = [start_node]
    all_poses = [gtsam.Pose3()]
    rel_poses = [gtsam.Pose3()]
    jump = 19
    for i in range(0, 3450, jump):
        print(i)
        pose_ck, ck, relative_marginal_covariance_mat = extract_relative_pose(database, stereo_k, i, min(i + jump, 3449))
        update_database_pose(database, current_transformation, i)
        R = pose_ck.rotation().matrix()
        t = pose_ck.translation()
        R_t = np.hstack((R, t[:, None]))
        current_transformation = compute_extrinsic_matrix(R_t, current_transformation)

        global_pose_ck = gtsam.Pose3(gtsam.Rot3(current_transformation[:, :3]), current_transformation[:, 3])
        all_poses.append(global_pose_ck)
        rel_poses.append(pose_ck)

        initial_poses[min(i + jump, 3449)] = current_transformation[:, 3]
        relative_pose = curr_pose.inverse().between(pose_ck)
        initialEstimate.insert(ck, relative_pose)
        next_node = Node(ck, {})
        next_node.add_neighbor(curr_node, relative_marginal_covariance_mat)
        all_nodes.append(next_node)
        curr_node.add_neighbor(next_node, relative_marginal_covariance_mat)
        factor = gtsam.BetweenFactorPose3(curr_symbol, ck, curr_pose.between(relative_pose), relative_marginal_covariance_mat)
        graph.add(factor)
        curr_pose = relative_pose
        curr_symbol = ck
        curr_node = next_node
    update_database_pose(database, current_transformation, 3449)
    error_before = graph.error(initialEstimate)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
    result = optimizer.optimize()
    error_after = optimizer.error()
    print("total error before optimization: ", error_before)
    print("total error after optimization: ", error_after)
    # marginals = gtsam.Marginals(graph, result)
    # plot_trajectory(1, result, scale=2, title="Locations as a 3D")
    # plt.show()
    # plot_trajectory(1, result, marginals=marginals, scale=2, title="Locations as a 3D include the Covariance of the locations")
    # plt.show()

    initial_poses = initial_poses.T
    plot_initial_pose(initial_poses[0], initial_poses[2])
    return all_poses, all_nodes, graph, optimizer, rel_poses, initial_poses, result


def plot_initial_pose(x, z):
    plt.scatter(x, z, c='blue', s=2)
    # plt.xlabel("z")
    # plt.ylabel("y")
    plt.title("trajecory of initial poses")
    plt.savefig("traj_initial.png")
    plt.clf()
