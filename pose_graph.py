from bundle_adjustment import *
from gtsam import gtsam, utils
from graph_utils import *
from typing import List


def get_absolute_pose_graph_error(estimated_ext_mat: List[gtsam.Pose3], num_of_cameras: int = 3450,
                                  jump: int = JUMP) -> None:  # estimated_ext_mat is all poses (from create_pose_graph)
    """
    This function plot the absolute pose graph estimation (without loop closure) error in X, Y, Z axis, the total
    error norm and the angle error.
    """
    needed_indices = [i for i in range(0, num_of_cameras, jump)] + [num_of_cameras - 1]
    real_locs = read_poses()[needed_indices]
    real_ext_mat = np.array(read_ground_truth_extrinsic_mat())[needed_indices]
    estimated_locations = []
    for mat in estimated_ext_mat:
        estimated_locations.append(mat.translation())
    estimated_locations = np.array(estimated_locations)

    absolute_estimation_error(real_locs.T, estimated_locations.T, real_ext_mat, estimated_ext_mat, jump=JUMP, estimation_type="pose_graph")


def extract_relative_pose(database: DataBase, stereo_k: gtsam.Cal3_S2Stereo, first: int, last: int,
                          current_transformation: FloatNDArray = np.hstack((np.eye(3), np.zeros((3, 1))))) -> Tuple[
    gtsam.Pose3, gtsam.symbol, gtsam.noiseModel.Gaussian, FloatNDArray]:  # q 6.1
    """
    This function extract the relative pose and the relative marginal covariance matrix between the frames first and last.
    """
    bundle_frames = list(range(first, last + 1))
    graph, result, frame_symbols, current_transformation = perform_bundle_window(database, stereo_k, bundle_frames, current_transformation)[3:]
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
    return pose_ck, ck, relative_marginal_covariance_mat, current_transformation


def create_pose_graph(database: DataBase, stereo_k: gtsam.StereoCamera) -> Tuple[
    List[gtsam.Pose3], List[Node], gtsam.NonlinearFactorGraph, gtsam.LevenbergMarquardtOptimizer, List[gtsam.Pose3], FloatNDArray, gtsam.Values]:
    """
    This function create a pose graph and perform bundle adjustment optimization of that graph.
    """
    initial_poses = np.zeros((3450, 3))
    current_transformation = np.hstack((np.eye(3), np.zeros((3, 1))))
    current_transformation2 = np.hstack((np.eye(3), np.zeros((3, 1))))
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
    jump = JUMP
    for i in tqdm(range(0, 3450, jump), desc="Creating pose graph", total=(math.ceil(3450 / jump))):
        pose_ck, ck, relative_marginal_covariance_mat, current_transformation2 = extract_relative_pose(database, stereo_k, i, min(i + jump, 3449),
                                                                                                       current_transformation2)
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
        # next_node.add_neighbor(curr_node, relative_marginal_covariance_mat)
        all_nodes.append(next_node)
        curr_node.add_neighbor(next_node, relative_marginal_covariance_mat)
        factor = gtsam.BetweenFactorPose3(curr_symbol, ck, curr_pose.between(relative_pose), relative_marginal_covariance_mat)
        graph.add(factor)
        curr_pose = relative_pose
        curr_symbol = ck
        curr_node = next_node
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


def plot_initial_pose(x, z) -> None:
    plt.scatter(x, z, c='blue', s=2)
    plt.title("trajectory of initial poses")
    plt.savefig("traj_initial.png")
    plt.clf()


if __name__ == '__main__':
    stereo_k = get_stereo_k()
    database = open_database()

    all_poses, all_nodes, graph, optimizer, rel_poses, initial_poses, result = create_pose_graph(database, stereo_k)
    get_absolute_pose_graph_error(all_poses, jump=JUMP)
