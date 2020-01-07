//************************************************************************
// compute_flow.cpp
// Computes OpenCV GPU Brox et al. [1] and Zach et al. [2] TVL1 Optical Flow
// Dependencies: OpenCV and Qt5 for iterating (sub)directories
// Author: Christoph Feichtenhofer -> Forked by Antonino Furnari -> Modified by Will Price (will.price@bristol.ac.uk)
// Institution: Graz University of Technology
// Email: feichtenhofer@tugraz
// Date: Nov. 2015
// [1] T. Brox, A. Bruhn, N. Papenberg, J. Weickert. High accuracy optical flow estimation based on a theory for warping. ECCV 2004.
// [2] C. Zach, T. Pock, H. Bischof: A duality based approach for realtime TV-L 1 optical flow. DAGM 2007.
//************************************************************************

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>
#include <queue>

#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

float MIN_SZ = 256;
float bound = 20;

// Global variables for gpu::BroxOpticalFlow
const float alpha_ = 0.197;
const float gamma_ = 50;
const float scale_factor_ = 0.8;
const int inner_iterations_ = 10;
const int outer_iterations_ = 77;
const int solver_iterations_ = 10;

#ifdef DEBUG
#define DEBUG_MSG(msg) do { std::cout << msg << std::endl; } while (0)
#else
#define DEBUG_MSG(msg)
#endif

static void convertFlowToImage(const Mat &flow_in, Mat &flow_out,
                               float lower_bound, float higher_bound) {
    #define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flow_in.rows; ++i) {
        for (int j = 0; j < flow_in.cols; ++j) {
            float x = flow_in.at<float>(i, j);
            flow_out.at<uchar>(i, j) = CAST(x, lower_bound, higher_bound);
        }
    }
    #undef CAST
}

int main( int argc, char *argv[] )
{
    GpuMat frame0GPU, frame1GPU, uGPU, vGPU;
    Mat current_frame, resized_frame;
    Mat frame0_rgb_, frame1_rgb_, frame0_rgb, frame1_rgb, frame0, frame1, rgb_out;
    Mat frame0_32, frame1_32, imgU, imgV;
    Mat motion_flow, flow_rgb;

    std::string input_folder, img_format, flow_format; //input video file name
    int gpuID = 0,
        type = 1,
        stride = 1,
        dilation = 1;

    const char* keys = {
                "{ h  | help     | false | print help message }"
                "{ g  | gpuID    |  0    | use this gpu}"
                "{ f  | type     |  1    | use this flow method [0:Brox, 1:TVL1]}"
                "{ s  | stride   |  1    | temporal stride (this subsamples the video)}"
                "{ d  | dilation |  1    | temporal dilation (1: use neighbouring frames, 2: skip one)}"
                "{ b  | bound    |  20   | maximum optical flow for clipping}"
                "{ z  | size     |  256  | minimum output size}"
    };

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Usage: compute_flow input_folder img_format flow_format [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printParams();
        cout << "Example: compute_flow P01_01 img_%07d.jpg flow_%s_%07d.jpg [options]" << endl;
        return 0;
    }

    if (argc > 2) {
        input_folder = argv[1];
        img_format = argv[2];
        flow_format = argv[3];

        gpuID = cmd.get<int>("gpuID");
        type = cmd.get<int>("type");
        stride = cmd.get<int>("stride");
        dilation = cmd.get<unsigned int>("dilation");
        bound = cmd.get<float>("bound");
        MIN_SZ = cmd.get<int>("size");
    } else {
        cout << "Not enough parameters!"<<endl;
        cout << "Usage: compute_flow input_folder img_formati flow_format [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printParams();
        cout << "Example: compute_flow P01_01 img_%07d.jpg flow_%s_%07d.jpg [options]" << endl;
        return 0;
    }
    int gpuCounts = cv::gpu::getCudaEnabledDeviceCount();
    cout << "Number of GPUs present " << gpuCounts << endl;

    cv::gpu::setDevice(gpuID);
    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

    cv::gpu::BroxOpticalFlow dflow(alpha_,gamma_,scale_factor_,inner_iterations_,outer_iterations_,solver_iterations_);
    cv::gpu::OpticalFlowDual_TVL1_GPU alg_tvl1;


    std::queue<cv::Mat> frame_queue;

    VideoCapture cap;
    try {
        cap.open(input_folder+'/'+img_format);
    } catch (std::exception& e) {
        std::cout << e.what() << '\n';
    }


    unsigned int width = 0, height = 0;
    float factor = 0;

    if( cap.isOpened() == 0 ) {
        cout << "Problem opening input file"<<endl;
        return -1; //exit with a -1 error
    }
    std::cout << "Processing folder " << input_folder << std::endl;

    unsigned int flow_frame_number = 1;

    assert(frame_queue.empty());
    unsigned int frame_index = 0;
    for (int i = 0; i < dilation; i++) {
        cap >> current_frame;
        frame_index++;
        if (current_frame.empty()) {
            std::cout << "Insufficient frames to fill frame queue--no flow computed." << std::endl;
            return 0;
        }

        factor = std::max<float>(MIN_SZ / current_frame.cols, MIN_SZ / current_frame.rows);

        width = std::floor(current_frame.cols * factor);
        width -= width%2;
        height = std::floor(current_frame.rows * factor);
        height -= height%2;

        resized_frame = cv::Mat(Size(width, height), CV_8UC3);

        cv::resize(current_frame, resized_frame, cv::Size(width, height), 0, 0, INTER_CUBIC);

        frame_queue.push(resized_frame.clone());
    }
    DEBUG_MSG("Queue filled");
    assert(frame_queue.size() == ((size_t) dilation));
    // len(frame_queue) == dilation

    unsigned int flow_index = 0;  // this is represents the position at which we *can* compute a flow field, although we don't
                         // always compute a flow field if temporal downsampling via a stride is specified.
    cap >> current_frame;
    frame_index++;
    while (!current_frame.empty()){
        DEBUG_MSG("Waiting on cap");
        DEBUG_MSG("Got frame " << frame_index);
        resized_frame = cv::Mat(Size(width, height), CV_8UC3);
        cv::resize(current_frame, resized_frame, cv::Size(width, height), 0, 0, INTER_CUBIC);

        DEBUG_MSG("Push frame " << frame_index);
        frame_queue.push(resized_frame.clone());

        DEBUG_MSG("Pop frame " << frame_index - dilation);
        //extract frame 0 from the front of the queue and pop
        frame0_rgb = frame_queue.front().clone();
        frame_queue.pop();
        DEBUG_MSG("Popped frame " << frame_index - dilation);

        if (flow_index % stride == 0) {
            // Allocate memory for the images
            flow_rgb = cv::Mat(Size(width, height), CV_8UC3);
            motion_flow = cv::Mat(Size(width, height), CV_8UC3);
            frame0 = cv::Mat(Size(width, height), CV_8UC1);
            frame1 = cv::Mat(Size(width, height), CV_8UC1);
            frame0_32 = cv::Mat(Size(width, height), CV_32FC1);
            frame1_32 = cv::Mat(Size(width, height), CV_32FC1);

            // Convert the image to grey and float
            cvtColor(resized_frame, frame1, CV_BGR2GRAY);
            frame1.convertTo(frame1_32, CV_32FC1, 1.0 / 255.0, 0);

            cvtColor(frame0_rgb, frame0, CV_BGR2GRAY);
            frame0.convertTo(frame0_32, CV_32FC1, 1.0 / 255.0, 0);

            switch (type) {
                case 0:
                    frame1GPU.upload(frame1_32);
                    frame0GPU.upload(frame0_32);
                    dflow(frame0GPU, frame1GPU, uGPU, vGPU);
                case 1:
                    DEBUG_MSG("Started frame upload");
                    frame1GPU.upload(frame1);
                    frame0GPU.upload(frame0);
                    DEBUG_MSG("Completed frame upload");
                    DEBUG_MSG("Started flow computation");
                    alg_tvl1(frame0GPU, frame1GPU, uGPU, vGPU);
                    DEBUG_MSG("Completed flow computation");
            }
            DEBUG_MSG("Complete flow computation");

            uGPU.download(imgU);
            vGPU.download(imgV);
            DEBUG_MSG("Downloaded flow");

            float min_u_f = -bound;
            float max_u_f = bound;

            float min_v_f = -bound;
            float max_v_f = bound;

            cv::Mat img_u(imgU.rows, imgU.cols, CV_8UC1);
            cv::Mat img_v(imgV.rows, imgV.cols, CV_8UC1);

            convertFlowToImage(imgU, img_u, min_u_f, max_u_f);
            convertFlowToImage(imgV, img_v, min_v_f, max_v_f);

            char *filename_x = new char[255];
            char *filename_y = new char[255];
            sprintf(filename_x, flow_format.c_str(), "x", flow_frame_number);
            sprintf(filename_y, flow_format.c_str(), "y", flow_frame_number);
            cv::imwrite(input_folder + "/" + string(filename_x), img_u);
            cv::imwrite(input_folder + "/" + string(filename_y), img_v);
            flow_frame_number++;
        }
        flow_index++;
        cout << "." << flush;

        cap >> current_frame;
        frame_index++;
    }

    cout << endl;
    return 0;
}


