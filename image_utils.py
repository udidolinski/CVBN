import numpy as np
from enum import IntEnum
from collections import namedtuple
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Union, Dict
from cv2 import KeyPoint, DMatch

FloatNDArray = NDArray[np.float64]


class FilterMethod(IntEnum):
    """
    This is an Enum for the filter methods (steps).
    """
    RECTIFICATION = 0
    QUAD = 1
    PNP = 2


class ImageColor(IntEnum):
    """
    This is an Enum for image colors
    """
    GRAY = 0
    RGB = 1


class Image:
    """
    This class represents the image of the frame including the key points, the descriptors and the inliers indices
    """

    def __init__(self, mat: NDArray[np.uint8], kp: Union[NDArray[KeyPoint], None] = None, desc: Union[NDArray[np.uint8], None] = None):
        self.mat = mat
        self.kps = kp
        self.desc = desc  # descriptors
        self.matches_kps_idx = np.zeros(1)
        self.rectified_kps_idx = np.zeros(1)
        self.quad_inliers_kps_idx = np.zeros(1)
        self.pnp_inliers_kps_idx = np.zeros(1)
        self.inliers_filter_funcs = [self._get_rectification_inliers_kps, self._get_quad_inliers_kps, self._get_pnp_inliers_kps]
        self.inliers_idx_filter_funcs = [self._get_rectification_inliers_idx, self._get_quad_inliers_kps_idx, self._get_pnp_inliers_kps_idx]
        self.outliers_filter_funcs = [self._get_rectification_outliers_kps, self._get_quad_outliers_kps, self._get_pnp_outliers_kps]
        self.outliers_idx_filter_funcs = [self._get_rectification_outliers_idx, self._get_quad_outliers_idx, self._get_pnp_outliers_idx]

    def is_rgb(self) -> bool:
        """
        This function returns True if the image is with 3 dimensions (in other words if it is an RGB image)
        :return:
        """
        return self.mat.ndim == 3

    def get_inliers_kps(self, filter_kind: FilterMethod) -> NDArray[KeyPoint]:
        """
        This function gets the inliers key points according to which FilterMethod provided
        :param filter_kind:
        :return:
        """
        return self.inliers_filter_funcs[filter_kind]()

    def get_inliers_kps_idx(self, filter_kind: FilterMethod) -> NDArray[np.int64]:
        """
        This function gets the inliers key points indices according to which FilterMethod provided
        :param filter_kind:
        :return:
        """
        return self.inliers_idx_filter_funcs[filter_kind]()

    def get_outliers_kps(self, filter_kind: FilterMethod) -> NDArray[KeyPoint]:
        """
        This function gets the inliers key points according to which FilterMethod provided
        :param filter_kind:
        :return:
        """
        return self.outliers_filter_funcs[filter_kind]()

    def get_outliers_kps_idx(self, filter_kind: FilterMethod) -> NDArray[np.int64]:
        """
        This function gets the inliers key points indices according to which FilterMethod provided
        :param filter_kind:
        :return:
        """
        return self.outliers_idx_filter_funcs[filter_kind]()

    def set_matches_kps_idx(self, matches_kps_idx):
        self.matches_kps_idx = matches_kps_idx

    def get_matches_kps_idx(self):
        return self.matches_kps_idx

    def set_rectification_inliers_kps(self, new_inliers_kp_idx: List[int]) -> None:
        """
        This function sets the rectification inliers indices
        :param new_inliers_kp_idx:
        :return:
        """
        self.rectified_kps_idx = new_inliers_kp_idx

    def _get_rectification_outliers_idx(self) -> NDArray[np.int64]:
        """
        This function gets the rectification outliers indices key points
        :return:
        """
        all_idx = {i for i in self.get_matches_kps_idx()}
        outliers_idx = list(all_idx - set(self.rectified_kps_idx))
        return np.array(outliers_idx)

    def _get_rectification_outliers_kps(self) -> NDArray[KeyPoint]:
        """
        This function gets the rectification outliers key points
        :return:
        """
        return self.kps[self._get_rectification_outliers_idx()]

    def _get_rectification_inliers_kps(self) -> NDArray[KeyPoint]:
        """
        This function gets the rectification inliers key points
        :return:
        """
        return self.kps[self.rectified_kps_idx]

    def _get_rectification_inliers_idx(self) -> NDArray[np.int64]:
        """
        This function gets the rectification inliers key points indices
        :return:
        """
        return self.rectified_kps_idx

    def _get_quad_inliers_kps(self) -> NDArray[KeyPoint]:
        """
        This function gets the quad inliers key points
        :return:
        """
        return self.kps[self.quad_inliers_kps_idx]

    def _get_quad_inliers_kps_idx(self) -> NDArray[np.int64]:
        """
        This function gets the quad inliers key points indices
        :return:
        """
        return self.quad_inliers_kps_idx

    def set_quad_inliers_kps_idx(self, quad_inliers_kps_idx: List[int]) -> None:
        """
        This function sets the quad inliers key points indices
        :param quad_inliers_kps_idx:
        :return:
        """
        self.quad_inliers_kps_idx = quad_inliers_kps_idx

    def _get_quad_outliers_idx(self):
        all_idx = {i for i in self._get_rectification_inliers_idx()}
        outliers_idx = list(all_idx - set(self._get_quad_inliers_kps_idx()))
        return np.array(outliers_idx)

    def _get_quad_outliers_kps(self) -> NDArray[KeyPoint]:
        """
        This function gets the quad outliers key points
        :return:
        """
        return self.kps[self._get_quad_outliers_idx()]

    def _get_pnp_inliers_kps_idx(self) -> NDArray[np.int64]:
        """
        This function gets the PNP inliers key points indices
        :return:
        """
        return self.pnp_inliers_kps_idx

    def _get_pnp_inliers_kps(self) -> NDArray[KeyPoint]:
        """
        This function gets the PNP inliers key points
        :return:
        """
        return self.kps[self._get_pnp_inliers_kps_idx()]

    def set_pnp_inliers_kps_idx(self, pnp_inliers_kps_idx: NDArray[np.int64]) -> None:
        """
        This function sets the PNP inliers key points indices
        :param pnp_inliers_kps_idx:
        :return:
        """
        self.pnp_inliers_kps_idx = [int(elem) for elem in np.array(self._get_quad_inliers_kps_idx())[pnp_inliers_kps_idx]]

    def _get_pnp_outliers_idx(self):
        all_idx = {i for i in self._get_quad_inliers_kps_idx()}
        outliers_idx = list(all_idx - set(self._get_pnp_inliers_kps_idx()))
        return outliers_idx

    def _get_pnp_outliers_kps(self) -> NDArray[KeyPoint]:
        """
        This function gets the quad outliers key points
        :return:
        """
        return self.kps[self._get_pnp_outliers_idx()]


class StereoPair:
    """
    This class represents a stereo pair of images including the matches between them, the frame index and
    the dictionary from left to right image (keypoint index of image 1 to keypoint in image 2 that was achieved by the match)
    """

    def __init__(self, left_image: Image, right_image: Image, idx: int, matches: Union[NDArray[DMatch], None] = None):
        self.left_image = left_image
        self.right_image = right_image
        self.idx = idx
        self.matches = matches
        self.__rectified_inliers_matches_idx = np.zeros(1)
        self.__quad_inliers_matches_idx = []
        self.pnp_inliers_matches_idx = []
        self.filter_matches_funcs = []
        self.left_right_kps_idx_dict = {}
        self.img1_kps_to_matches_idx = np.zeros(1)
        self.img2_kps_to_matches_idx = np.zeros(1)
        # self.filter_method = FilterMethod.NO_FILTER

    def set_images_kps_to_matches_idx(self, img1_kps_to_matches_idx, img2_kps_to_matches_idx) -> None:
        self.img1_kps_to_matches_idx = img1_kps_to_matches_idx
        self.img2_kps_to_matches_idx = img2_kps_to_matches_idx

    def get_rectified_inliers_matches(self) -> NDArray[DMatch]:
        """
        This function gets the rectification inliers matches of the pair
        :return:
        """
        return self.matches[self.__rectified_inliers_matches_idx]

    def set_rectified_inliers_matches_idx(self, rectified_inliers_matches_idx: NDArray[np.int64]) -> None:
        """
        This function sets the rectification inliers matches indices of the pair
        :param rectified_inliers_matches_idx:
        :return:
        """
        self.__rectified_inliers_matches_idx = rectified_inliers_matches_idx

    def get_rectified_outliers_matches_idx(self) -> List[int]:
        """
        This function gets the rectification outliers matches indices of the pair
        :return:
        """
        return list(set(range(len(self.matches))) - set(self.__rectified_inliers_matches_idx))

    def set_quad_inliers_matches_idx(self, quad_inliers_matches_idx: List[int]) -> None:
        """
        This function sets the rectification outliers matches indices of the pair
        :param quad_inliers_matches_idx:
        :return:
        """
        self.__quad_inliers_matches_idx = self.__rectified_inliers_matches_idx[quad_inliers_matches_idx]

    def get_quad_inliers_matches_idx(self) -> List[int]:
        """
        This function gets the inlier quad matches indices
        :return:
        """
        return self.__quad_inliers_matches_idx

    def get_quad_inliers_matches(self) -> NDArray[DMatch]:
        """
        This function gets the inlier quad matches
        :return:
        """
        return self.matches[self.get_quad_inliers_matches_idx()]

    def get_pnp_outliers_matches_idx(self) -> List[int]:
        """
        This function gets the inlier PNP matches indices
        :return:
        """
        return list(set(self.__rectified_inliers_matches_idx) - set(self.__rectified_inliers_matches_idx[self.pnp_inliers_matches_idx]))

    def set_left_right_kps_idx_dict(self, left_right_kps_idx_dict: Dict[int, int]) -> None:
        """
        This function sets the dictionary from left to right image (keypoint index of image 1 to keypoint in image 2 that was achieved by the match)
        :param left_right_kps_idx_dict:
        :return:
        """
        self.left_right_kps_idx_dict = left_right_kps_idx_dict

    def get_left_right_kps_idx_dict(self) -> Dict[int, int]:
        """
        This function gets the dictionary from left to right image (keypoint index of image 1 to keypoint in image 2 that was achieved by the match)
        :return:
        """
        return self.left_right_kps_idx_dict


class Quad:
    """
    This class represents a pair of consecutive StereoPair's objects (represents frames i and i+1) including the matches between the
    left images of the two pairs, the relative transformation matrix between the left image of pair 1 to the left image of pair 2 and
    the dictionary that maps between key points from the left image of pair 2 to the left image of pair 1 that matched
    """

    def __init__(self, stereo_pair1: StereoPair, stereo_pair2: StereoPair, left_left_matches: Union[NDArray[DMatch], None] = None):
        self.stereo_pair1 = stereo_pair1
        self.stereo_pair2 = stereo_pair2
        self.left_left_matches = left_left_matches
        self.relative_trans = np.zeros(1)  # T from pair 1 to pair 2
        self.left_left_kps_idx_dict = {}  # key is keypoint of left image of pair 2 and value is keypoint of left image of pair 1 (of left left matches)

    def set_left_left_matches(self, left_left_matches: NDArray[DMatch]) -> None:
        """
        This function sets the the matches between the left images of the two pairs
        :param left_left_matches:
        :return:
        """
        self.left_left_matches = left_left_matches

    def set_left_left_kps_idx_dict(self, left_left_kps_idx_dict: Dict[int, int]) -> None:
        """
        This function sets the dictionary that maps between key points from the left image of pair 2 to the left image of pair 1 that matched
        :param left_left_kps_idx_dict:
        :return:
        """
        self.left_left_kps_idx_dict = left_left_kps_idx_dict

    def get_left_left_kps_idx_dict(self) -> Dict[int, int]:
        """
        This function gets the dictionary that maps between key points from the left image of pair 2 to the left image of pair 1 that matched
        :return:
        """
        return self.left_left_kps_idx_dict

    def set_relative_trans(self, transform: FloatNDArray) -> None:
        """
        This function sets the relative transformation matrix between the left image of pair 1 to the left image of pair 2
        :param transform:
        :return:
        """
        self.relative_trans = transform

    def get_relative_trans(self) -> FloatNDArray:
        """
        This function gets the relative transformation matrix between the left image of pair 1 to the left image of pair 2
        :return:
        """
        return self.relative_trans


TrackInstance = namedtuple("TrackInstance", ["x_l", "x_r", "y"])


class Track:
    """
    This class represent track.
    """

    def __init__(self, track_id: int, kp_idx: int, pair_id: int):
        """
        Constructur.
        :param track_id:
        :param kp_idx: the last keypoint index the track is.
        :param pair_id: the last pair id track this track appears in.
        """
        self.track_id = track_id
        self.last_kp_idx = kp_idx
        self.last_pair = pair_id
        self.frame_ids = []
        self.track_instances = []

    def set_last_pair_id(self, pair_id: int) -> None:
        """
        This function sets the last pair id (stereo pair of images) this track appears in.
        :param pair_id:
        :return:
        """
        self.last_pair = pair_id

    def set_last_kp_idx(self, kp_idx: int) -> None:
        """
        This function sets the last keypoint index (in the last stereo pair of images) this track appears is.
        :param kp_idx:
        :return:
        """
        self.last_kp_idx = kp_idx

    def __gt__(self, other: int) -> bool:
        """
        This function is a comparator between 2 tracks (one is bigger it has more frames he appears in).
        :param other:
        :return:
        """
        return len(self.frame_ids) > other


class Frame:
    """
    This class represent frame which includes frame id (index), inliers percentage
    and the transformation matrix from frame 0 to the current frame id
    """

    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.track_ids = []
        self.inliers_percentage = 0
        self.matches_num = 0
        self.transformation_from_zero = np.zeros(1)
        self.transformation_from_after_bundle = np.zeros(1)

    def get_matches_num(self) -> int:
        return self.matches_num

    def set_matches_num(self, matches_num) -> None:
        self.matches_num = matches_num

    def get_inliers_percentage(self) -> float:
        """
        This function gets the inliers percentage
        :return:
        """
        return self.inliers_percentage

    def set_inliers_percentage(self, inliers_percentage: float) -> None:
        """
        This function sets the inliers percentage
        :param inliers_percentage:
        :return:
        """
        self.inliers_percentage = inliers_percentage

    def set_transformation_from_zero(self, transformation_from_zero: FloatNDArray) -> None:
        """
        This function sets the transformation matrix from frame 0 to the current frame
        :param transformation_from_zero:
        :return:
        """
        self.transformation_from_zero = transformation_from_zero

    def set_transformation_from_zero_after_bundle(self, transformation_from_zero_bundle: FloatNDArray) -> None:
        self.transformation_from_after_bundle = transformation_from_zero_bundle

    def get_transformation_from_zero(self) -> FloatNDArray:
        """
        This function gets the transformation matrix from frame 0 to the current frame
        :return:
        """
        return self.transformation_from_zero

    def get_transformation_from_zero_bundle(self) -> FloatNDArray:
        return self.transformation_from_after_bundle


class DataBase:
    """
    This class holds the tracks and frames objects.
    """

    def __init__(self, tracks: List[Track], frames: List[Frame]):
        self.tracks = tracks
        self.frames = frames

    def get_num_of_tracks(self) -> int:
        """
        This function return the number of tracks in the database.
        """
        return len(self.tracks)

    def get_num_of_frames(self) -> int:
        """
        This function return the number of frames in the database.
        """
        return len(self.frames)

    def get_mean_track_length(self) -> float:
        """
        This function return the mean track length of the tracks in the database.
        """
        track_sum = 0
        for track in self.tracks:
            track_sum += len(track.frame_ids)
        return track_sum / len(self.tracks)

    def get_max_track_length(self) -> int:
        """
        This function return the maximum length of track in the database.
        """
        return len(max(self.tracks, key=lambda x: len(x.frame_ids)).frame_ids)

    def get_min_track_length(self) -> int:
        """
        This function return the minimum length of track in the database.
        """
        return len(min(self.tracks, key=lambda x: len(x.frame_ids)).frame_ids)

    def get_mean_number_of_frame_links(self) -> np.float64:
        """
        This function return the mean number of frame links of track in the database.
        """
        res = []
        for i in range(len(self.frames) - 1):
            res.append(len((set(self.frames[i].track_ids) & set(self.frames[i + 1].track_ids))))
        return np.mean(res)

    def create_connectivity_graph(self) -> None:
        """
        This function creates and plot the tracks connectivity graph.
        """
        res = []
        for i in range(len(self.frames) - 1):
            res.append(len((set(self.frames[i].track_ids) & set(self.frames[i + 1].track_ids))))
        plt.figure(figsize=(12, 4))
        plt.plot(res)
        mean_outgoing_tracks = round(np.mean(res))
        plt.plot([mean_outgoing_tracks for _ in range(len(res))], label=f'Mean={mean_outgoing_tracks}', linestyle='--')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('outgoing tracks')
        plt.title('connectivity graph')
        plt.savefig("connectivity_graph.png")
        plt.clf()

    def inliers_percentage_graph(self) -> None:
        """
        This function creates and plot the inliers percentage graph.
        """
        percentage = [frame.inliers_percentage for frame in self.frames]
        plt.figure(figsize=(12, 4))
        plt.plot(percentage)
        mean_inliers_persentage_per_frame = round(np.mean(percentage), 2)
        plt.plot([mean_inliers_persentage_per_frame for _ in range(len(percentage))], label=f'Mean={mean_inliers_persentage_per_frame}', linestyle='--')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('frame')
        plt.ylabel('inliers percentage')
        plt.title('percentage graph')
        plt.savefig("percentage_graph.png")
        plt.clf()

    def create_track_length_histogram_graph(self) -> None:
        """
        This function creates and plot the track length histogram graph.
        """
        track_length = [len(track.frame_ids) for track in self.tracks]
        bins = np.arange(max(track_length) + 1)
        hist = np.histogram(track_length, bins)[0]
        plt.figure(figsize=(8, 4))
        plt.plot(hist)
        plt.xlabel('track length')
        plt.ylabel('track #')
        plt.title('track length histogram')
        plt.savefig("track_length_histogram.png")
        plt.clf()

    def num_of_matches_per_frame_graph(self):
        """
        This function creates and plot the matches number per fame graph.
        """
        matches_num = [frame.matches_num for frame in self.frames]
        plt.figure(figsize=(12, 4))
        plt.plot(matches_num)
        mean_matches_per_frame = round(np.mean(matches_num))
        plt.plot([mean_matches_per_frame for _ in range(len(matches_num))], label=f'Mean={mean_matches_per_frame}', linestyle='--')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('matches number')
        plt.title('matches number graph')
        plt.savefig("matches_number_graph.png")
        plt.clf()
