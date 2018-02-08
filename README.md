GPU based optical flow extraction from videos
=================================================
Forked from https://github.com/feichtenhofer/gpu_flow

### Features
The tool allows to extract optical flow from video. By default, the tool resizes the input video so that its height matches 256 pixels and aspect ratio remains the same. The tool creates a video where x and y optical flow images are stored side by side. Optical flow is obtained by clipping large displacement. Other options are available modifying the source code (see https://github.com/feichtenhofer/gpu_flow).

Usage:
    ./compute_flow [OPTION] videofile.avi
Output:
    videofile_flow.avi

### Dependencies:
 * [OpenCV 2.4] (http://opencv.org/downloads.html)
 * [cmake] (https://cmake.org/)

### Installation
First, build opencv with gpu support. To do so, download opencv 2.4.x sources from https://opencv.org/releases.html. Unzip the downloaded archive, then enter the opencv folder and issue the following commands:

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

### Usage
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
