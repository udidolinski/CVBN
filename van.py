import random
import sys
from loop_closure import *
if __name__ == '__main__':
    mode = "all"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    num_of_cameras = 3450
    real_ext_mat = read_ground_truth_extrinsic_mat()
    real_locs = read_poses()
    needed_indices = [i for i in range(0, num_of_cameras, JUMP)] + [num_of_cameras - 1]
    real_locs_keyframes = real_locs[needed_indices].T
    stereo_k = get_stereo_k()

    # database = create_database()
    # save_database(database)
    database = open_database("database_neww")

    if mode == "initial_estimate" or mode == "all":
        print("calculating initial estimate...")
        est_locs = get_pnp_locations(database)
        real_locs = read_poses().T
        pnp_ext_mat = get_pnp_extrinsic_mat(database)
        plot_trajectury(est_locs[0], est_locs[2], real_locs[0], real_locs[2], "initial_estimate_result")
        absolute_estimation_error(real_locs, est_locs, real_ext_mat, pnp_ext_mat, "PnP")
        new_reprojection_error(database, False)  # reprojection and factor error
        relative_estimation_error(100, real_ext_mat, pnp_ext_mat, "PnP")
        relative_estimation_error(300, real_ext_mat, pnp_ext_mat, "PnP")
        relative_estimation_error(800, real_ext_mat, pnp_ext_mat, "PnP")
        print("done calculating initial estimate.")

    if mode == "bundle_adjustment" or mode == "all":
        print("start performing bundle adjustment...")
        perform_bundle(database)
        est_locs_bundle = np.zeros((num_of_cameras, 3))
        for i in range(num_of_cameras):
            est_locs_bundle[i] = transform_rt_to_location(database.frames[i].get_transformation_from_zero_bundle())
        est_ext_mat_bundle = get_bundle_extrinsic_mat(database)
        plot_trajectury(est_locs_bundle.T[0], est_locs_bundle.T[2], real_locs.T[0], real_locs.T[2], "bundle_adjustment_results")
        absolute_estimation_error(real_locs.T, est_locs_bundle.T, real_ext_mat, est_ext_mat_bundle, "bundle_adjustment")
        new_reprojection_error(database, True)  # reprojection and factor error
        relative_estimation_error(100, real_ext_mat, est_ext_mat_bundle, "bundle_adjustment")
        relative_estimation_error(300, real_ext_mat, est_ext_mat_bundle, "bundle_adjustment")
        relative_estimation_error(800, real_ext_mat, est_ext_mat_bundle, "bundle_adjustment")
        print("done performing bundle adjustment.")

    if mode == "pose_graph" or mode == "all":
        print("start creating pose graph...")
        all_poses, all_nodes, graph, optimizer, rel_poses, l2, result = create_pose_graph(database, stereo_k)
        plot_trajectory_from_result(result, "pose_graph_results", 3450)
        get_absolute_pose_graph_error(all_poses)
        print("done creating pose graph.")

    if mode == "loop_closure" or mode == "all":
        print("start performing loop closure...")
        all_poses, all_nodes, graph, optimizer, rel_poses, l2, result = create_pose_graph(database, stereo_k)
        res, new_graph = detect_loop_closure_candidates(all_poses, all_nodes, graph, database, stereo_k, result, rel_poses)
        plot_trajectory_from_result(res, "loop_closure_results", 3450)
        get_absolute_loop_closure_error(res)
        marginals_before = gtsam.Marginals(graph, result)
        marginals_after = gtsam.Marginals(new_graph, res)
        plot_uncertainty_graph(marginals_before, marginals_after)
        print("done performing loop closure.")

    if mode == "database":
        present_statistics(database)
