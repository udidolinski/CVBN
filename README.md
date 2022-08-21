## Computer Vision Based Navigation


**Implementation SLAM (Simultaneous localization and mapping). The system runs on [KITTI](http://www.cvlibs.net/datasets/kitti/) stereo data.**<br/>
**We used image processing and optimization techniques to estimate the vehicle path and the geometric structure of the world.**



<b> Overview</b>:

- [Consensus Matching](#consensus-matching) – robust feature tracking for moving stereo cameras.

- [Initial Estimate of trajectory](#initial-estimate-of-trajectory) - finding initial estimate to trajectory using feature tracking from the previous step.  
Triangulation and [RANSAC](#ransac), with [PnP](#pnp) as the inner model.  

- [Creating Database](#creating-database) -  creating tracks database.

- [Bundle Adjustment Optimization](#bundle-adjustment-optimization) - performing bundle adjustment optimization in order to fix the initial estimate.

- [Pose Graph](#pose-graph) - creating pose graph.

- [Loop Closure](#loop-closure) - performing loop closure in order to fix the trajectory.


<details>
  <summary><b>How to run the program</b>: </summary>

  Notice that because we used [GTSAM](https://gtsam.org) you can only run the program on linux/macOS environment.
  <br/>
  <br/>
  <b> Requirements:</b> <br/>
  linux/mac OS environment, installing the following libraries: gtsam, numpy, openCV.  <br/>
  <br/>
  After you make sure you have all the requirements, preform the following steps: <br/>

  1. Make sure you have KITTI dataset. Move the dataset to the directory CVBN (step 2).  <br/>
  2. Clone the repository into your local environment:  <br/>
  ```bash
  $ git clone https://github.com/udidolinski/CVBN
  ```
  3. Change the working directory:
  ```bash
  $ cd CVBN
  ```
  4. Run the program using the following commands:  

| command  | result  |
| :--- |:---|
| `van.py initial_estimate`       | Perform initial Estimate, saves the trajectory to initial_estimate_results.png        |
| `van.py bundle_adjustment`      | Perform bundle adjustment, saves the trajectory to bundle_adjustment_results.png      |
| `van.py pose_graph`             | create pose graph, saves the trajectory to pose_graph_results.png                     |
| `van.py loop_closure`           | Perform loop closure, saves the trajectory to loop_closure_results.png                |
| `van.py all`                    | Perform all the steps above                                                           |
| `van.py database`               | Print statistics information of database.db                                           |


example:  

```bash
$ python3 van.py loop_closure
```
</details>

<br/>
<br/>

# All the details:
we started with detect, extract and match key points in our stereo pair images.
We used SIFT to detect key points and found matches using brute force matcher.
In order to save this information and access it easily we build the class [Image](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/image_utils.py#L29) that holds the image, key points and the key points descriptors and the class [StereoPair](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/image_utils.py#L199) that holds two Images and their matches.
After preforming the matching we wanted to remove the outliers, because the dataset consists of stereo images we could use a pattern that stereo images has (y coordinates of matching key points need to be the same).  
We choose to use threshold of 0.5 pixel, so only the matches that differ in the y coordinates in less or equal to 0.5 will survive this test. <br/>

At this point we preform a few sanity checks to make sure we going in the right direction.
So we presented the matches, and preform triangulation. This two checks showed us that we are doing good but there are still some outliers that we didn't removed (for example we got from the triangulation 3d point that the z value is negative).
In order to remove the remaining outliers we used consensus matching.

## Consensus Matching
At this point we have StereoPair object for each image pair that holds the matches between them.
The first step in consensus matching is to match those key points in the previous left image and the current left image:  
<br/>
<br/>
 <img src="/images/image1.png"  width=50% height=50%>  

 This method helped us removing almost any outliers that some how survived BF matching and the pattern test. So after performing those steps we left with the key-points that were matched on all four images.  
 We used ransac, with PnP as the inner model to filter the inlier, and now we have our inliers keypoints.
 You can access those key-points using:
 ```python
 Quad.StereoPair.Image.get_inliers_kps(FilterMethod.QUAD)
 ```

## Initial Estimate of trajectory
We wanted to estimate the initial position of each camera in the global coordinate system.  
The camera pose consists of 6 degrees-of-freedom (DOF) which are made up of the rotation (roll, pitch, and yaw) and 3D translation of the camera with respect to the world.  
Let's review some definitions:
- <b>extrinsic camera matrix</b> - transformation matrix from the world coordinate system to the camera coordinate system. The matrix composed from 3x3 rotation matrix and 3x1 translation vector: [R|t].  
- <b>intrinsic camera matrix</b> - transformation matrix that converts points from the camera coordinate system to the pixel coordinate system. In order to get the the intrinsic camera matrix you can use the following code:  
```python
k, m1, m2 = read_cameras()
left_intrinsic_camera_matrix, right_intrinsic_camera_matrix = k @ m1, k @ m2
```
The difference between the cameras is 0.53 centimeters on the x axis.   

<br/>
In order to locate the camera position in the global coordinate system we can use the extrinsic matrix:  
Lets say the camera location is (x y z), we know that the camera location in the camera coordinate system is (0 0 0). So if we will solve the following equations system: [R|t] (x y z 1)^T = 0 we will find the camera position in the global coordinate system.  
<br/>
<br/>
 <img src="/images/image2.png"  width=60% height=60%>  

So now we want to find the extrinsic camera matrix for each frame in order to calculate the camera position, we will do that using PnP.  

PnP - problem of estimating the pose of a calibrated camera given a set of n 3D points in the world and their corresponding 2D projections in the image.  

RANSAC (Random sample consensus) -  iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers.  
The parameters we choose for ransac algorithm are:  
  - number of samples = 4
  - probability of success = 0.99
  - initial outliers percentage = 0.85   
<br/>  

To find the cameras relative locations we used RANSAC, with PnP as the inner model.
For each iteration we created QUAD that consist of two consecutive images, solve PnP for n=4 by randomly choose 4 key-points (that survived the consensus matching), perform triangulation in order to find their locations in the world, and sent those locations as well as the key-points pixels to get estimation of the extrinsic camera matrix.
```python
succeed, rvec, tvec = cv2.solvePnP(points_3d, image_points, k, None, flags=flag)
R_t = rodriguez_to_mat(rvec, tvec)
camera_location = transform_rt_to_location(R_t)
```
During the run of the firs loop in the algorithm, we changed the outliers percentage to be the smallest we found from all the model we tried so far, and calculated the number of iteration each time we found model that more points agreed on.  
After choosing the model with the maximal number of inliers, we perform another Loop and in each iteration we estimate model from all inliers, found the new inliers and repeat for 5 iteration.
By the end of the RANSAC we have in our quad the relative transformation from pair_1 to pair_2.  
Reference to code: [PnP](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/initial_estimate.py#L430), [RANSAC](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/initial_estimate.py#L522).  

To get the initial estimate of the cameras position and as a result the initial estimate of the trajectory we used RANSAC on each two consecutive images, and composed their relative transformation to get their locations from the initial (0 0 0).  
Explain about how we composed those transformations:  
For three cameras A, B, C, such that A extrinsic matrix is [I|0], T_a_b is the transformation A -> B and T_b_c is the transformation B -> C the transformation A -> C is:  
<br/>  
 <img src="/images/image3.png"  width=40% height=40%>

so the location of camera C in the global coordinate system we be calculated using:
```python
transformation_a_to_c = compute_extrinsic_matrix(T_a_b, T_b_c)
c_location = transform_rt_to_location(transformation_a_to_c)
```
 Finally we got all the cameras locations and plot the trajectory compared to ground truth:  
 ```python
 g_t = read_poses().T  # ground truth
 cam_loc_estimate = trajectory()  # initial estimate
 plot_trajectury(cam_loc_estimate[0], cam_loc_estimate[2], g_t[0], g_t[2])  
```
The result we got:  
<br/>
<img src="/images/image4.png"  width=60% height=60%>

Finding the initial estimate took 27 minutes, the buttle neck is to detect, extract and match the key-points as well as preforming consensus matching. In order to avoid doing those actions in the future we created database.  

## Creating Database
Our [database](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/image_utils.py#L446) consist list of Frames and list of Tracks.

[Tracks](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/image_utils.py#L346) are feature points we found in the previous steps (meaning they found in two consecutive images), that now we are elaborating the tracking to be to as many consecutive images instead of just two.  
Each track has a id, list of Frames its shows in, and the pixel (x_r, x_l, y) and the length of tracks is the number of frames it shows in.  

For each [Frame](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/image_utils.py#L389) we save the id, and list of the Tracks ids that showed in the frame.  
<br/>
<img src="/images/image5.png"  width=70% height=70%>

The tracking process:  
we iterate over all the frames, for each frame we took the key points we have from the previous steps and search for it in the next frames, then we added it to the database.
we got 383,603 tracks and the mean track length is 4.78.  
Track example:  
<br/>
<img src="/images/image6.png"  width=60% height=60%>

Useful code:
```python
database = create_database(start_frame_id, end_frame_id: int, start_track_id,tracks, frames)
save_database(database)  # save the data to the file database.db
database = open_database()
```

## Bundle Adjustment Optimization
Now that we have an initial estimate of the vehicle trajectory and our dataset we are ready to take the next step and optimize the trajectory in order to get an accurate estimation of the trajectory.  

**Projection** is taking a world coordinate (3d points = (x y z 1)) and project it on the camera. The projection done by multiplay the intrinsic matrix, extrinsic matrix and the 3d point.
```python
perform_transformation_3d_points_to_pixels(extrinsic_mat, intrinsic_mat, point3d=(x y z 1))
```                                  

**Reprojection error**  is the distance (L2 norm) between the projection of 3d point and the pixel we know it located.
For each track we can calculate the projection error by taking a 3d point from the last frame the track shows in and project it to all the other frames using the extrinsic matrix of the frame.
The optimization process will use the projection error to perform the optimization.
<br/>

<img src="/images/image7.png">



From now on we used GTSAM optimization library to handle the mathematical details of the process.

**Bundle Adjustment:**  
Minimizing the reprojection error between the image locations of observed image points, which is expressed as the sum of squares of a large number of nonlinear, real-valued functions. Thus, the minimization is achieved using nonlinear least-squares algorithms.

[Bundle Adjustment window](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/bundle_adjustment.py#L506) - performing local Bundle Adjustment on a small window consisting of consecutive frames.
Each bundle window starts and ends in special frames we call keyframes.  
<img src="/images/bundle_win.png"  width=70% height=70%>

For each window we building a graph. first we add the graph a constraint to set the first key frame position to (0,0,0), then we use the tracking database we have, in order to build the rest of the graph.  
Each track in the last key frame will be a constraint in the graph - means we iterate on all the tracks we found in the last key frame and perform triangulation in order to get 3d point (we call in landmark).  
Now, we iterate on all the other frame in the window, for each frame we add the initial estimate of the position (that we saved in the database) and iterate on all the tracks (of the last frame) and add a factor to our graph, the factor has the landmark (3d point) and the location of the landmark in the frame (we have it in the database), this is our constraint.  
<img src="/images/bundle.png"  width=70% height=70%>

```python
initialEstimate = gtsam.Values()
graph = gtsam.NonlinearFactorGraph()
initialEstimate.insert(start_key_frame, gtsam.Pose3())  # set the first key frame position to (0,0,0)  
```  
```python
# adding the constraints
initialEstimate.insert(frame_symbol, frame_pose)
factor = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(*location), stereo_model, frame_symbol, landmark, stereo_k)
graph.add(factor)
```  

After adding all the constraints we perform optimization and get the optimized location of the last frame. pay attention that the locations we get from the bundle are relative and we need to concatenate them in order to get the global position.

```python
# optimize
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
result = optimizer.optimize()
```  

At the end of the [bundle adjustment](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/bundle_adjustment.py#L584) we have an optimized trajectory of the vehicle path:  
<img src="/images/image8.png"  width=60% height=60%>



## Pose Graph
We build a pose graph in order to perform a final optimization in the next step.
The pose graph is used to keep a concise summary of the trajectory. It consists of the poses of some of the key frames and the relative nonlinear pose constraints between them. Since it contains no reprojection constraints and only a subset of the frame poses it is a very concise representation of the entire trajectory.  

<img src="/images/pose.png"  width=65% height=65%>


In order to build a factor graph that represents the pose graph of the keyframes we iterate on all keyframes.  
For each keyframe we extract the pose and the relative covariance of each consecutive keyframe from the bundle optimization, transform the pose to global and calculate the relative pose between the previous keyframe and the one we extracted. we added between factor to the graph.

```python
pose_ck, ck_symbol, relative_marginal_covariance_mat = extract_relative_pose(database, stereo_k, i, min(i+jump, 3449))
relative_pose = curr_pose.inverse().between(pose_ck)
factor = gtsam.BetweenFactorPose3(curr_symbol, ck, curr_pose.between(relative_pose), relative_marginal_covariance_mat)
graph.add(factor)
```  

The relative covariance present the uncertainty of the motion and we will use it the next step.


<img src="/images/locations_as_a_3d_include_the_covariance_of_the_locations.png" width=65% height=65%>

## Loop Closure

Now that we have a pose graph we want to use it to recognize if the vehicle is returned to a previously visited region of the world.
When we find such connection to a past frame, we will use it to add a Loop Closure constraint to the pose graph, thus greatly reducing the drift of the trajectory estimation.
This will help us limiting the growth in uncertainty, and perform final optimization of the trajectory.  

**Detect Loop Closure candidates:**

In order to find if frame c_n and c_i (i<n) are loop closure candidates we started with finding the shortest path between them and sum the covariances along the path to get an estimate of the relative covariance.  
Because we cant iterate on gtsam.NonlinearFactorGraph object we created dummy graph that represent the pose graph. The graph build such that we have a root [node](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/graph_utils.py#L48), and each node holds his neighbors and their covariances.  
The nodes in the graph are the nodes in the pose graph (the key frames pose), and the edges are the connections between key frames.  
We used [USC](https://github.com/udidolinski/CVBN/blob/be65bc29c4c4f79f3d64cdbfcd3b2187cda66de8/graph_utils.py#L100) (uniform cost search) in order to find the shortest path. The edges weight was the relative covariance between the nodes.

After we have the relative covariance we calculated Mahalanobis distance between the nodes and decided a 500,000 threshold.  

For each pair of candidates that pass the mahalanobis distance threshold we perform consensus matching.
If we found pair that has at least 60% of inliers we finds the relative pose between them and adding a loop closure edge (between factor) to the graph.  

Example of a successful consensus matching:
| ![](/images/2394_.png) |
|:--:|
| *frame 2393, inliers in orange* |

| ![](/images/3344_.png) |
|:--:|
| *frame 3344, inliers in orange* |

After we insert new factor to the graph we perform an optimization.

```python
def detect_loop_closure_candidates(all_poses: List[gtsam.Pose3], all_nodes: List[Node], pose_graph: gtsam.NonlinearFactorGraph, database: DataBase, stereo_k: gtsam.Cal3_S2Stereo, optimizer: gtsam.LevenbergMarquardtOptimizer, rel_poses: List[gtsam.Pose3]):
   for c_n_idx in range(1, len(all_nodes)):
       for c_i_idx in range(c_n_idx):
           cov, success = get_relative_covariance(all_nodes[c_n_idx], all_nodes[c_i_idx])
           rel_pos = all_poses[c_i_idx].inverse().between(all_poses[c_n_idx].inverse())
           mahalanobis_dist = mahalanobis_distance(cov, rel_pos)
           if c_n_idx-c_i_idx>=40 and mahalanobis_dist < MAHALANOBIS_DISTANCE_TEST:
               inliers_percentage, inliers_locs = consensus_matching(c_n_idx, c_i_idx)
               if inliers_percentage >= CONSENSUS_MATCHING_THRESHOLD:
                   relative_pose, covariance = small_bundle(rel_poses[c_i_idx], rel_poses[c_n_idx], [c_i_idx, c_n_idx], database, stereo_k, inliers_locs)
                   # adding the new edge
                   all_nodes[c_i_idx].add_neighbor(all_nodes[c_n_idx], covariance)
                   all_nodes[c_n_idx].add_neighbor(all_nodes[c_i_idx], covariance)
                   factor = gtsam.BetweenFactorPose3(gtsam.symbol('x', c_i_idx), gtsam.symbol('x', c_n_idx), relative_pose, covariance)
                   pose_graph.add(factor)
                   # optimize
                   result = optimizer.optimize()

```  

Loop Closure conclusion:
We found 5 successful loop closures, we plot the trajectory with the covariances after each loop closure edge insertion and optimization.  
Its easy to see that the uncertainty got better each time (also comparing to the result after the first bundle).
The trajectory with the covariances after each edge insertion:
| ![](/images/after_frames_1520_76_traj_3d_cov.png) |
|:--:|
| *optimization after inserting 76-1520 edge* |

| ![](/images/after_frames_3344_2394_traj_3d_cov.png) |
|:--:|
| *optimization after inserting 2394-3344 edge* |

| ![](/images/after_frames_3439_437_traj_3d_cov.png) |
|:--:|
| *optimization after inserting 437-3439 edge* |


Also the trajectory became closer to the ground truth, the final trajectory:
![](/images/final_traj.png)



The error in meters after each step:
![](/images/absolute_pnp_estimation_error.png)  
![](/images/absolute_bundle_adjustment_estimation_error.png)  
![](/images/absolute_loop_closure_estimation_error.png)  

Those graphs shows that after each optimization step the absolute error bacome smaller, which means the estimation gets closer to the real tarjectory.   
