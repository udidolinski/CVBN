import matplotlib.pyplot as plt

from initial_estimate import *
from typing import Iterator
import pickle
import random
from tqdm import tqdm


def create_quads(start_frame_id: int, end_frame_id: int) -> Iterator[Quad]:
    """
    This function is a generator fro creating the Quad's objects
    :param start_frame_id: the frame id to the first quad that will be created (first quad includes the frames start_frame_id and start_frame_id+1)
    :param end_frame_id: the frame id to the last id that will be created (last quad includes the frames end_frame_id-1 and end_frame_id)
    :return:
    """
    k = read_cameras()[0]
    curr_stereo_pair2 = None
    for i in range(start_frame_id, end_frame_id):
        # print(i)
        current_quad = ransac(i, i + 1, k, curr_stereo_pair2)[0]
        yield current_quad
        curr_stereo_pair2 = current_quad.stereo_pair2


def gen_track_id(index: int) -> Iterator[int]:
    """
    This is generator for creating track id's
    :param index: which track id to start from (in case we extend the database we want to start generating id's from where we stopped before)
    :return:
    """
    while True:
        yield index
        index += 1


def get_left_image_pair1_kps_idx(quad: Quad, idx: int) -> int:
    """
    This function is for getting
    the key point index in the left image of pair 1 from
    the key point index in the left image of pair 2 that were matched.
    left image pair 2 -> left image pair 1
    :param quad:
    :param idx:
    :return:
    """
    return quad.left_left_kps_idx_dict[idx]


def get_right_image_pair_kps_idx(pair: StereoPair, pair1_left_img_kp_idx: int) -> int:
    """
    This function is for getting
    the key point index in the right image of a stereo pair
    the key point index in the left image of the same stereo pair
    left image -> right image
    :param pair:
    :param pair1_left_img_kp_idx:
    :return:
    """
    return pair.get_left_right_kps_idx_dict()[pair1_left_img_kp_idx]


def create_track(latest_tracks: List[Track], frames: List[Frame], inliers_idx: NDArray[np.int64], quad: Quad, track_id_gen: Iterator[int]) -> None:
    """
    This function create new track or continuing a track.
    :param latest_tracks: all tracks that created.
    :param frames: all frames that created.
    :param inliers_idx: the inliers of the left frame in pair2 of quad.
    :param quad:
    :param track_id_gen:
    """
    pair1_left_img_kp_idx = {get_left_image_pair1_kps_idx(quad, idx) for idx in inliers_idx}
    kp_to_track_dict = {latest_tracks[track_id].last_kp_idx: track_id for track_id in frames[-2].track_ids}
    kp_to_track_dict = {kp_idx: kp_to_track_dict[kp_idx] for kp_idx in pair1_left_img_kp_idx if kp_idx in kp_to_track_dict}
    for idx in inliers_idx:
        track_index = -1
        if get_left_image_pair1_kps_idx(quad, idx) in kp_to_track_dict:
            # This case the track already exists from 2nd to last frame (frames[-2])
            track_id = kp_to_track_dict[get_left_image_pair1_kps_idx(quad, idx)]
            track_index = kp_to_track_dict[get_left_image_pair1_kps_idx(quad, idx)]
        else:
            # This case we need to create a new track
            frame_id = quad.stereo_pair1.idx
            track_id = next(track_id_gen)
            frames[frame_id].track_ids.append(track_id)
            new_track = Track(track_id, idx, quad.stereo_pair1.idx)
            latest_tracks.append(new_track)
            pair1_left_img_kp_idx = get_left_image_pair1_kps_idx(quad, idx)
            pair1_right_img_kp_idx = get_right_image_pair_kps_idx(quad.stereo_pair1, pair1_left_img_kp_idx)
            kp_l = quad.stereo_pair1.left_image.kps[pair1_left_img_kp_idx]
            x_l = kp_l.pt[0]
            x_r = quad.stereo_pair1.right_image.kps[pair1_right_img_kp_idx].pt[0]
            y = kp_l.pt[1]
            latest_tracks[-1].track_instances.append(TrackInstance(x_l, x_r, y))
            latest_tracks[-1].frame_ids.append(frame_id)

        right_frame_id = quad.stereo_pair2.idx
        frames[right_frame_id].track_ids.append(track_id)
        kp_l = quad.stereo_pair2.left_image.kps[idx]
        right_x_l = kp_l.pt[0]
        right_x_r = quad.stereo_pair2.right_image.kps[get_right_image_pair_kps_idx(quad.stereo_pair2, idx)].pt[0]
        right_y = kp_l.pt[1]

        latest_tracks[track_index].set_last_pair_id(right_frame_id)
        latest_tracks[track_index].set_last_kp_idx(idx)
        latest_tracks[track_index].track_instances.append(TrackInstance(right_x_l, right_x_r, right_y))
        latest_tracks[track_index].frame_ids.append(right_frame_id)


def create_database(start_frame_id: int = 0, end_frame_id: int = 3449, start_track_id: int = 0,
                    tracks: Union[List[Track], None] = None, frames: Union[List[Frame], None] = None) -> DataBase:
    """
    This function creates a database of tracks.
    :param start_frame_id: first frame to store the tracks from.
    :param end_frame_id: last frame to store the tracks.
    :param start_track_id: track id to start from.
    :param tracks:
    :param frames:
    :return: The database
    """

    if not tracks:
        tracks = []
    if not frames:
        frames = [Frame(start_frame_id)]
        frames[-1].set_transformation_from_zero(np.hstack((np.eye(3), np.zeros((3, 1)))))
    gen = gen_track_id(start_track_id)
    for i, quad in tqdm(enumerate(create_quads(start_frame_id, end_frame_id)), desc="Creating DataBase", total=end_frame_id - start_frame_id + 1):
        current_transformation = frames[-1].get_transformation_from_zero()
        percentage = len(quad.stereo_pair1.left_image.get_inliers_kps(FilterMethod.RECTIFICATION)) / len(quad.stereo_pair1.matches)
        frames[-1].set_inliers_percentage(round(percentage, 2))
        frames[-1].set_matches_num(len(quad.stereo_pair1.matches))
        transformation_i_to_i_plus_1 = quad.get_relative_trans()
        transformation_0_to_i_plus_1 = compute_extrinsic_matrix(current_transformation, transformation_i_to_i_plus_1)
        frames.append(Frame(quad.stereo_pair2.idx))
        frames[-1].set_transformation_from_zero(transformation_0_to_i_plus_1)
        create_track(tracks, frames, quad.stereo_pair2.left_image.get_inliers_kps_idx(FilterMethod.PNP), quad, gen)
    percentage = len(quad.stereo_pair2.left_image.get_inliers_kps(FilterMethod.RECTIFICATION)) / len(quad.stereo_pair2.matches)
    frames[-1].set_inliers_percentage(round(percentage, 2))
    frames[-1].set_matches_num(len(quad.stereo_pair2.matches))
    return DataBase(tracks, frames)


def extend_database(database: DataBase, end_frame_id: int) -> DataBase:
    """
    This function extend the database from the last entry to end_frame_id.
    """
    start_frame_id = len(database.frames) - 1
    start_track_id_gen = len(database.tracks)
    new_database = create_database(start_frame_id, end_frame_id, start_track_id_gen, database.tracks, database.frames)
    return new_database


def save_database(database: DataBase, name: str = "database") -> None:
    """
    This function save the database to a file.
    """
    with open(f"{name}.db", "wb") as file:
        pickle.dump(database, file)


def open_database(database_name: str = "database") -> DataBase:
    """
    This function open the database file and return thr database object.
    """
    with open(f"{database_name}.db", "rb") as file:
        database = pickle.load(file)
    return database


def get_all_track_ids(frame_id: int, database: DataBase) -> List[int]:
    """
    This function return all the track_id that appear on a given frame_id.
    """
    return database.frames[frame_id].track_ids


def get_all_frame_ids(track_id: int, database: DataBase) -> List[int]:
    """
    This function return all the frame_id that are part of a given track_id.
    """
    return database.tracks[track_id].frame_ids


def get_feature_locations(frame_id: int, track_id: int, database: DataBase) -> TrackInstance:
    """
    This function return Feature locations of track TrackId on both left and right images as a triplet (x_l, x_r, y).
    """
    for frame, track_instance in zip(database.tracks[track_id].frame_ids, database.tracks[track_id].track_instances):
        if frame == frame_id:
            return track_instance


def result_image_size(image_shape: Tuple[int, int, int], x: int, y: int) -> Tuple[int, int, int, int]:
    """
    This function find the image size to present track
    :param image_shape:
    :param x: x track location
    :param y: y track location
    :return: The image size to display
    """
    x1 = max(0, x - 50) if image_shape[1] - x >= 50 else image_shape[1] - 100
    x2 = x1 + 100
    y1 = max(0, y - 50) if image_shape[0] - y >= 50 else image_shape[0] - 100
    y2 = y1 + 100
    return x1, x2, y1, y2


def display_track(database: DataBase, random_track: Track = None) -> None:
    """
    This function display a random track of length > 9.
    :param database:
    """
    if not random_track:
        tracks_bigger_than_10 = list(np.array(database.tracks)[np.array(database.tracks) > 9])
        random_track = random.choice(tracks_bigger_than_10)
    frames_idx = get_all_frame_ids(random_track.track_id, database)
    for frame in frames_idx:
        track_instance = get_feature_locations(frame, random_track.track_id, database)
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
        cv2.imwrite(f"Output Image {frame}.png", res)
    # cv2.waitKey(0)


def read_camera_matrices(first_index: int = 0, last_index: int = 3450) -> Iterator[FloatNDArray]:
    """
    This function is a generator for creating the true transformation matrices of the cameras
    (that transforms from the first camera to the the current camera)
    :param first_index:
    :param last_index:
    :return:
    """
    with open(os.path.join(POSES_PATH, '00.txt')) as f:
        for l in f.readlines()[first_index:last_index]:
            l = l.split()
            extrinsic_matrix = np.array([float(i) for i in l])
            extrinsic_matrix = extrinsic_matrix.reshape(3, 4)
            yield extrinsic_matrix


def calculate_norm(a: FloatNDArray, b: FloatNDArray) -> Union[float, FloatNDArray]:
    """
    This functions calculates the L2 norm between 2 vectors (numpy arrays)
    :param a:
    :param b:
    :return:
    """
    return np.linalg.norm(a - b)


def point_reprojection_error(extrinsic_matrix: FloatNDArray, k: FloatNDArray, p4d: FloatNDArray, location: List[float]) -> \
        Union[float, FloatNDArray]:
    """
    This function calculate the reprojection error of p4d by calculating the norm between the "real" pixel (location)
    of this point and the projection of the point.
    :param extrinsic_matrix:
    :param k:
    :param p4d:
    :param location: [x,y]
    :return: reprojection error
    """
    projected_pixel_2d = perform_transformation_3d_points_to_pixels(extrinsic_matrix, k, p4d).squeeze()
    return calculate_norm(projected_pixel_2d, np.array(location))


def reprojection(database: DataBase) -> None:
    """
    This function calculate and plot the reprojection error (the distance between the projection and the tracked feature
     location on that camera)
    :param database: Tracks database
    """
    left_error = []
    right_error = []
    tracks_bigger_than_10 = list(np.array(database.tracks)[np.array(database.tracks) > 9])
    random_track = random.choice(tracks_bigger_than_10)
    k, m1, m2 = read_cameras()
    P = k @ m1
    Q = k @ m2
    track_location = random_track.track_instances[-1]
    cv_p4d = cv2.triangulatePoints(P, Q, (track_location.x_l, track_location.y), (track_location.x_r, track_location.y)).squeeze()
    cv_p3d = cv_p4d[:3] / cv_p4d[3]
    cv_p3d2 = transform_rt_to_location(next(read_camera_matrices(random_track.frame_ids[-1], random_track.frame_ids[-1] + 1)), cv_p3d)
    p4d = np.hstack((cv_p3d2, 1))[:, None]
    for i, extrinsic_matrix in enumerate(read_camera_matrices(random_track.frame_ids[0], random_track.frame_ids[-1] + 1)):
        location = random_track.track_instances[i]
        left_error.append(point_reprojection_error(extrinsic_matrix, k, p4d, [location.x_l, location.y]))

        right_extrinsic_matrix = compute_extrinsic_matrix(extrinsic_matrix, m2)
        right_error.append(point_reprojection_error(right_extrinsic_matrix, k, p4d, [location.x_r, location.y]))

    plot_projection_error('reprojection error', left_error, right_error)


def plot_projection_error(title: str, left_error: List[float], right_error: List[float] = None, file_name: str="preojection_error") -> None:
    """
    This functions plots the reprojection error of the left images and the right images (or only the left images)
    :param title:
    :param left_error:
    :param right_error:
    :return:
    """
    plt.figure(figsize=(15, 5))
    plt.plot(left_error, label="left error")
    if right_error is not None:
        plt.plot(right_error, label="right error")
    plt.legend()
    plt.xlabel('distance from reference')
    plt.ylabel('projection error (pixels)')
    plt.title(title)
    plt.savefig(f"{file_name}.png")
    plt.clf()
    # plt.show()


def plot_reprojection_compared_to_factor_error(x_reprojection: List[float], y_factor: List[float]) -> None:
    """
    This function plots the reprojection error compared to the factor error
    :param x_reprojection:
    :param y_factor:
    :return:
    """
    plt.plot(x_reprojection, y_factor)
    plt.xlabel('reprojection error')
    plt.ylabel('factor error')
    plt.title("factor error as a function of reprojection error")
    plt.savefig("factor_to_reproj_error.png")
    plt.clf()


def present_statistics(database: DataBase) -> None:
    """
    This function presents statistics about the database as shown below
    :param database:
    :return:
    """
    print("Total number of tracks: ", database.get_num_of_tracks())
    print("Number of frames: ", database.get_num_of_frames())
    print("Mean track length: ", database.get_mean_track_length())
    print("Min track length: ", database.get_min_track_length())
    print("Max track length: ", database.get_max_track_length())
    print("Mean number of frame links: ", database.get_mean_number_of_frame_links())
    # display_track(database)
    database.create_connectivity_graph()
    database.inliers_percentage_graph()
    database.create_track_length_histogram_graph()
    # database.num_of_matches_per_frame_graph()  # todo
    reprojection(database)


if __name__ == '__main__':
    database = create_database()
    save_database(database, "database_neww")
