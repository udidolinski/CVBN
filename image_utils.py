import numpy as np
from enum import IntEnum
from collections import namedtuple
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Union, Dict
from cv2 import KeyPoint, DMatch

FloatNDArray = NDArray[np.float64]


class FilterMethod(IntEnum):
    RECTIFICATION = 0
    QUAD = 1
    PNP = 2


class ImageColor(IntEnum):
    GRAY = 0
    RGB = 1


class Image:
    def __init__(self, mat: NDArray[np.uint8], kp: Union[NDArray[KeyPoint], None] = None, desc: Union[NDArray[np.uint8], None] = None):
        self.mat = mat
        self.kps = kp
        self.desc = desc  # descriptors
        self.inliers_kps_idx = np.zeros(1)
        self.quad_inliers_kps_idx = np.zeros(1)
        self.pnp_inliers_kps_idx = np.zeros(1)
        self.inliers_filter_funcs = [self.get_rectification_inliers_kps, self.get_quad_inliers_kps, self.get_pnp_inliers_kps]
        self.inliers_idx_filter_funcs = [self.get_rectification_inliers_idx, self.get_quad_inliers_kps_idx, self.get_pnp_inliers_kps_idx]
        self.outliers_filter_funcs = 1

    def is_rgb(self) -> bool:
        return self.mat.ndim == 3

    def get_inliers_kps(self, filter_kind: FilterMethod) -> NDArray[KeyPoint]:
        return self.inliers_filter_funcs[filter_kind]()

    def get_inliers_kps_idx(self, filter_kind: FilterMethod) -> NDArray[np.int64]:
        return self.inliers_idx_filter_funcs[filter_kind]()

    def get_ouliers_kps(self) -> NDArray[KeyPoint]:
        return self.kps[self.get_outliers_idx()]

    def get_rectification_inliers_kps(self) -> NDArray[KeyPoint]:
        return self.kps[self.inliers_kps_idx]

    def get_rectification_inliers_idx(self) -> NDArray[np.int64]:
        return self.inliers_kps_idx

    def set_inliers_kps(self, new_inliers_kp_idx: List[int]) -> None:
        self.inliers_kps_idx = new_inliers_kp_idx

    def get_outliers_idx(self) -> NDArray[np.int64]:
        all_idx = {i for i in range(len(self.kps))}
        return np.array(all_idx - self.inliers_kps_idx)

    def get_rectification_outliers_kps(self) -> NDArray[KeyPoint]:
        return self.kps[self.get_outliers_idx()]

    def get_quad_inliers_kps(self) -> NDArray[KeyPoint]:
        return self.kps[self.quad_inliers_kps_idx]

    def get_quad_inliers_kps_idx(self) -> NDArray[np.int64]:
        return self.quad_inliers_kps_idx

    def set_quad_inliers_kps_idx(self, quad_inliers_kps_idx: List[int]) -> None:
        self.quad_inliers_kps_idx = quad_inliers_kps_idx

    def get_pnp_inliers_kps_idx(self) -> NDArray[np.int64]:
        return self.pnp_inliers_kps_idx

    def get_pnp_inliers_kps(self) -> NDArray[KeyPoint]:
        return self.get_quad_inliers_kps()[self.pnp_inliers_kps_idx]

    def set_pnp_inliers_kps_idx(self, pnp_inliers_kps_idx: NDArray[np.int64]) -> None:
        self.pnp_inliers_kps_idx = pnp_inliers_kps_idx


class StereoPair:
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
        # self.filter_method = FilterMethod.NO_FILTER

    def get_rectified_inliers_matches(self) -> NDArray[DMatch]:
        return self.matches[self.__rectified_inliers_matches_idx]

    def set_rectified_inliers_matches_idx(self, rectified_inliers_matches_idx: NDArray[np.int64]) -> None:
        self.__rectified_inliers_matches_idx = rectified_inliers_matches_idx

    def get_rectified_outliers_matches_idx(self) -> List[int]:
        return list(set(range(len(self.matches))) - set(self.__rectified_inliers_matches_idx))

    def set_quad_inliers_matches_idx(self, quad_inliers_matches_idx: List[int]) -> None:
        self.__quad_inliers_matches_idx = quad_inliers_matches_idx

    def get_quad_inliers_matches(self) -> NDArray[DMatch]:
        return self.matches[self.__rectified_inliers_matches_idx[self.__quad_inliers_matches_idx]]

    def get_quad_inliers_matches_idx(self) -> NDArray[np.int64]:
        return self.__rectified_inliers_matches_idx[self.__quad_inliers_matches_idx]

    def get_pnp_outliers_matches_idx(self) -> List[int]:
        return list(set(self.__rectified_inliers_matches_idx) - set(self.__rectified_inliers_matches_idx[self.pnp_inliers_matches_idx]))

    def set_left_right_kps_idx_dict(self, left_right_kps_idx_dict: Dict[int,int]) -> None:
        self.left_right_kps_idx_dict = left_right_kps_idx_dict

    def get_left_right_kps_idx_dict(self) -> Dict[int,int]:
        return self.left_right_kps_idx_dict


class Quad:
    def __init__(self, stereo_pair1: StereoPair, stereo_pair2: StereoPair, left_left_matches: Union[NDArray[DMatch], None] = None):
        self.stereo_pair1 = stereo_pair1
        self.stereo_pair2 = stereo_pair2
        self.left_left_matches = left_left_matches
        self.relative_trans = np.zeros(1)  # T from pair 1 to pair 2
        self.left_left_kps_idx_dict = {}  # key is keypoint of left image of pair 2 and value is keypoint of left image of pair 1 (of left left matches)

    def set_left_left_matches(self, left_left_matches: NDArray[DMatch]) -> None:
        self.left_left_matches = left_left_matches

    def set_left_left_kps_idx_dict(self, left_left_kps_idx_dict: Dict[int, int]) -> None:
        self.left_left_kps_idx_dict = left_left_kps_idx_dict

    def get_left_left_kps_idx_dict(self) -> Dict[int, int]:
        return self.left_left_kps_idx_dict

    def set_relative_trans(self, transform: FloatNDArray) -> None:
        self.relative_trans = transform

    def get_relative_trans(self) -> FloatNDArray:
        return self.relative_trans


TrackInstance = namedtuple("TrackInstance", ["x_l", "x_r", "y"])


class Track:
    """
    This class represent track.
    """
    def __init__(self, track_id: int, kp_idx:int, pair_id: int):
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
        self.last_pair = pair_id

    def set_last_kp_idx(self, kp_idx: int) -> None:
        self.last_kp_idx = kp_idx

    def __gt__(self, other: int) -> bool:
        return len(self.frame_ids) > other


class Frame:
    """
    This class represent frame.
    """
    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.track_ids = []
        self.inliers_percentage = 0
        self.transformation_from_zero = np.zeros(1)
        self.transformation_from_after_bundle = np.zeros(1)

    def get_inliers_percentage(self) -> float:
        return self.inliers_percentage

    def set_inliers_percentage(self, inliers_percentage: float) -> None:
        self.inliers_percentage = inliers_percentage

    def set_transformation_from_zero(self, transformation_from_zero: FloatNDArray) -> None:
        self.transformation_from_zero = transformation_from_zero

    def set_transformation_from_zero_after_bundle(self, transformation_from_zero_bundle: FloatNDArray) -> None:
        self.transformation_from_after_bundle = transformation_from_zero_bundle

    def get_transformation_from_zero(self) -> FloatNDArray:
        return self.transformation_from_zero

    def get_transformation_from_zero_bundle(self) -> FloatNDArray:
        return self.transformation_from_after_bundle


class DataBase:
    """
    This class hold the tracks and frames objects.
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
        sum = 0
        for track in self.tracks:
            sum += len(track.frame_ids)
        return sum/len(self.tracks)

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

    def get_mean_number_of_frame_links(self) -> float:
        """
        This function return the mean number of frame links of track in the database.
        """
        return len(self.tracks)/len(self.frames)

    def create_connectivity_graph(self) -> None:
        """
        This function creates and plot the tracks connectivity graph.
        """
        res = []
        for i in range(len(self.frames)-1):
            res.append(len((set(self.frames[i].track_ids)&set(self.frames[i+1].track_ids))))
        plt.figure(figsize=(12, 5))
        plt.plot(res)
        plt.xlabel('frame')
        plt.ylabel('outgoing tracks')
        plt.title('connectivity graph')
        plt.show()

    def inliers_percentage_graph(self) -> None:
        """
        This function creates and plot the inliers percentage graph.
        """
        percentage = [frame.inliers_percentage for frame in self.frames]
        plt.figure(figsize=(12, 5))
        plt.plot(percentage)
        plt.ylim(0, 1)
        plt.xlabel('frame')
        plt.ylabel('inliers percentage')
        plt.title('percentage graph')
        plt.show()

    def create_track_length_histogram_graph(self) -> None:
        """
        This function creates and plot the track length histogram graph.
        """
        track_length = [len(track.frame_ids) for track in self.tracks]
        bins = np.arange(max(track_length)+1)
        hist = np.histogram(track_length, bins)[0]
        plt.figure(figsize=(12, 5))
        plt.plot(hist)
        plt.xlabel('track length')
        plt.ylabel('track #')
        plt.title('track length histogram')
        plt.show()