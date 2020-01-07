GPU based optical flow extraction from videos
=================================================
Forked from https://github.com/feichtenhofer/gpu_flow by Antonino Furnari

[![Build Status](https://travis-ci.org/dl-container-registry/furnari-flow.svg?branch=master)](https://travis-ci.org/dl-container-registry/furnari-flow)
[![Docker Hub](https://img.shields.io/badge/hosted-dockerhub-22b8eb.svg)](https://hub.docker.com/r/willprice/furnari-flow/)
[![Singularity Hub](https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg)](https://singularity-hub.org/collections/575)


## Usage

We support running via docker and singularity.

### Docker

* Ensure you're running
  [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker) as this software is
  GPU accelerated.
* Pull the docker image: 
  ```
  $ docker pull willprice/furnari-flow
  ```
* Dump out frames from the video you wish to compute flow for:
  ```sh
  $ mkdir my_video; ffmpeg -i my_video.mp4 my_video/img_%06d.jpg
  ```
* Compute the flow using `furnari-flow`:
  ```sh
  $ mkdir my_video_flow
  $ docker 
      --runtime=nvidia \
      --rm \
      --mount "type=bind,source=$PWD/my_video,target=/input" \
      --mount "type=bind,source=$PWD/my_video_flow,target=/output" \
      --mount "type=bind,source=$HOME/.nv,target=/cache/nv" \
      willprice/furnari-flow \
      img_%06d.jpg
  $ ls my_video_flow
  u v
  $ ls my_video_flow/u
  img_0000001.jpg
  img_0000002.jpg
  ...
  ```

### Details

The software assumes that all video frames have been extracted in a directory. Files should be named according to some pattern, e.g., `img_%07d.jpg`. The software will put flow files in the same directory using a provided filename pattern, e.g., `flow_%s_%07d.jpg`, where the %s will be subsituted with "x" for the x flows and "y" for the y flows. For example, if DIR is a directory containing 4 images:

DIR:

 * `img_0000001.jpg`
 * `img_0000002.jpg`
 * `img_0000003.jpg`
 * `img_0000004.jpg`

the command `compute_flow DIR img_%07d.jpg flow_%s_%07d.jpg` will read the images in order and compute optical flows. The content of DIR will be as follows after the execution of the command:

DIR:

 * `img_0000001.jpg`
 * `img_0000002.jpg`
 * `img_0000003.jpg`
 * `img_0000004.jpg`
 * `flow_x_0000001.jpg`
 * `flow_x_0000002.jpg`
 * `flow_x_0000003.jpg`
 * `flow_x_0000004.jpg`
 * `flow_y_0000001.jpg`
 * `flow_y_0000002.jpg`
 * `flow_y_0000003.jpg`
 * `flow_y_0000004.jpg`

where `flow_x_{n}.jpg` is the x flow computed between `img_{n}.jpg` and `img_{n+1}.jpg` (if no dilation is used - see help).

## Build

You only need to build this software if you intend on tweaking the source, otherwise you
should just use the pre-built docker images.

### Dependencies:
 * [OpenCV 2.4](http://opencv.org/downloads.html)
 * [cmake](https://cmake.org/)


### Installation
First, build opencv with gpu support. To do so, download opencv 2.4.x sources
from https://opencv.org/releases.html. Unzip the downloaded archive, then enter
the opencv folder and issue the following commands:

 * `mkdir build`
 * `cd build`
 * `cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..`
 * `make`

Then clone the current repository and enter the `compute_flow_video` folder. Type:

 * `export OpenCV_DIR=path_to_opencv_build_directory`
 * `mkdir build`
 * `cd build`
 * `cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..`
 * `make`

