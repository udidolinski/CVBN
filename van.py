import sys
from loop_closure import *


if __name__ == '__main__':
    """
    Usage:
    initial_estimate - calculating initial estimate, plots the results and the relevant graphs.
    bundle_adjustment - performing bundle adjustment optimization, plots the results and the relevant graphs.
    pose_graph - creating a pose graph, plots the results and the relevant graphs.
    loop_closure - performing loop closure, plots the final results and the relevant graphs.
    all - performing all the steps above and plots the results and the relevant graphs for each step.
    database - present statistics information about thr database..
    """
    mode = "all"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    sequence_len = [100, 300, 800]
    num_of_cameras = 3450
    real_ext_mat = read_ground_truth_extrinsic_mat()
    real_locs = read_poses()
    needed_indices = [i for i in range(0, num_of_cameras-1, JUMP)] + [num_of_cameras - 1]
    real_locs_keyframes = real_locs[needed_indices].T
    real_locs = real_locs.T
    stereo_k = get_stereo_k()

    # database = create_database()
    # save_database(database)
    database = open_database("database")

    if mode == "initial_estimate" or mode == "all":
        print("calculating initial estimate...")
        est_locs = get_pnp_locations(database)
        real_locs = read_poses().T
        pnp_ext_mat = get_pnp_extrinsic_mat(database)
        plot_trajectury(est_locs[0], est_locs[2], real_locs[0], real_locs[2], "initial_estimate_result")
        absolute_estimation_error(real_locs, est_locs, real_ext_mat, pnp_ext_mat, "PnP")
        new_reprojection_error(database, False)  # reprojection and factor error
        for i in sequence_len:
            relative_estimation_error(i, real_ext_mat, pnp_ext_mat, "PnP")
        print("done calculating initial estimate.")

    if mode == "bundle_adjustment" or mode == "all":
        print("start performing bundle adjustment...")
        perform_bundle(database)
        est_locs_bundle = np.zeros((num_of_cameras, 3))
        for i in range(num_of_cameras):
            est_locs_bundle[i] = transform_rt_to_location(database.frames[i].get_transformation_from_zero_bundle())
        est_ext_mat_bundle = get_bundle_extrinsic_mat(database)
        plot_trajectury(est_locs_bundle.T[0], est_locs_bundle.T[2], real_locs[0], real_locs[2], "bundle_adjustment_results")
        absolute_estimation_error(real_locs, est_locs_bundle.T, real_ext_mat, est_ext_mat_bundle, "bundle_adjustment")
        new_reprojection_error(database, True)  # reprojection and factor error
        for i in sequence_len:
            relative_estimation_error(i, real_ext_mat, est_ext_mat_bundle, "bundle_adjustment")
        print("done performing bundle adjustment.")

    if mode == "pose_graph" or mode == "all":
        print("start creating pose graph...")
        all_poses, all_nodes, graph, optimizer, rel_poses, l2, result = create_pose_graph(database, stereo_k)
        marginals_before = gtsam.Marginals(graph, result)
        plot_uncertainty_graph(marginals_before, "Pose_Graph")
        plot_trajectory_from_result(result, "pose_graph_results", num_of_cameras)
        get_absolute_pose_graph_error(all_poses)
        print("done creating pose graph.")

    if mode == "loop_closure" or mode == "all":
        print("start performing loop closure...")
        all_poses, all_nodes, graph, optimizer, rel_poses, l2, result = create_pose_graph(database, stereo_k)
        marginals_before = gtsam.Marginals(graph, result)
        plot_uncertainty_graph(marginals_before, "Pose_Graph")
        res, new_graph = detect_loop_closure_candidates(all_poses, all_nodes, graph, database, stereo_k, result, rel_poses)
        init_locs = get_pnp_locations(database).T
        plot_all_results(result, res, init_locs)
        plot_trajectory_from_result(res, "loop_closure_results", num_of_cameras)
        get_absolute_loop_closure_error(res)
        marginals_after = gtsam.Marginals(new_graph, res)
        plot_uncertainty_graph(marginals_after, "Loop_Closure")
        print("done performing loop closure.")

    if mode == "database":
        print("The database created with the following parameters: ")
        print(f"- Y pattern threshold: {DEVIATION_THRESHOLD}")
        print(f"- AKAZE detector and BFMatcher with Hamming norm")
        print(f"- Ransac success probability: {RANSAC_SUCCESS_PROB}")
        print(f"- PnP threshold: {PNP_THRESHOLD}")
        present_statistics(database)
