# RRelation3DMOT: Exploiting Deep Affinity for 3D Multi-Object Tracking from View Aggregation
created by Can Chen, Luca Zanotti Fragonara, Antonios Tsourdos from Cranfield University

[[Paper]](https://arxiv.org/abs/2011.12850)

# Overview
We would like to leverage the advantages of LIDAR and camera sensors by proposing a deep neural network architecture for 3D MOT. Autonomous systems need to localize and track surrounding objects in 3D space for safe motion planning. As a result, 3D multi-object tracking (MOT) plays a vital role in autonomous navigation. Most MOT methods use a tracking-by-detection pipeline, which includes object detection and data association processing. However, many approaches detect objects in 2D RGB sequences for tracking, which is lack of reliability when localizing objects in 3D space. Furthermore, it is still challenging to learn discriminative features for temporally-consistent detection in different frames, and the affinity matrix is normally learned from independent object features without considering the feature interaction between detected objects in the different frames. To settle these problems, We firstly employ a joint feature extractor to fuse the 2D and 3D appearance features captured from both 2D RGB images and 3D point clouds respectively, and then propose a novel convolutional operation, named RelationConv, to better exploit the correlation between each pair of objects in the adjacent frames, and learn a deep affinity matrix for further data association. 

# Requirement
* [Pytorch](https://pytorch.org/)
