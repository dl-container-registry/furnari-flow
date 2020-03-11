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
#include <queue>

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

float MIN_SZ = 256;
float bound = 20;

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
    Mat current_frame, resized_frame;

    std::string input_folder, img_format, flow_format; //input video file name
    int gpuID = 0,
        type = 1,
        stride = 1,
        dilation = 1;

    const char* keys = {
                "{ @input_dir             | <none>            | Path to input directory containing frames }"
                "{ @frame_template        | frame_%010d.jpg   | Printf template for input frame filename }"
                "{ @flow_template         | flow_%s_%010d.jpg | Printf template for output flow frame filename }"

                "{ h help                 |                   | Print help message }"
                "{ g gpu-id               | 0                 | Use this GPU }"
                "{ f type                 | 1                 | Flow method [0:Brox, 1:TVL1] }"
                "{ s stride               | 1                 | Temporal stride (this subsamples the video) }"
                "{ d dilation             | 1                 | Temporal dilation (1: use neighbouring frames, 2: skip one) }"
                "{ b bound                | 20                | maximum optical flow for clipping }"
                "{ z size                 | 256               | Minimum output size }"

                "{ tvl1-epsilon           | 0.01              | TVL1 epsilon parameter }"
                "{ tvl1-lambda            | 0.15              | TVL1 lambda parameter }"
                "{ tvl1-tau               | 0.25              | TVL1 tau parameter }"
                "{ tvl1-theta             | 0.3               | TVL1 tau parameter }"
                "{ tvl1-gamma             | 0.0               | TVL1 (u-v)^2 tightness parameter }"
                "{ tvl1-iterations        | 300               | TVL1 number of iterations }"
                "{ tvl1-scale-step        | 0.8               | TVL1 pyramid scale factor }"
                "{ tvl1-nscales           | 5                 | TVL1 number of scales }"
                "{ tvl1-warps             | 5                 | TVL1 number of warps }"
                "{ tvl1-use-initial-flow  | false             | Whether to reuse flow fields between frames }"

                "{ brox-alpha             | 0.197             | Brox smoothness parameter }"
                "{ brox-gamma             | 50                | Brox gradient constancy parameter }"
                "{ brox-scale-factor      | 0.8               | Brox pyramid scale factor }"
                "{ brox-inner-iterations  | 10                | Brox number of inner iterations }"
                "{ brox-outer-iterations  | 77                | Brox number of outer iterations }"
                "{ brox-solver-iterations | 10                | Brox number of solver iterations }"
    };

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.has("help"))
    {
        cout << "Usage: compute_flow input_folder img_format flow_format [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
        cout << "Example: compute_flow P01_01 img_%07d.jpg flow_%s_%07d.jpg [options]" << endl;
        return 0;
    }

    if (cmd.check()) {
        input_folder = argv[1];
        img_format = argv[2];
        flow_format = argv[3];

        gpuID = cmd.get<int>("gpu-id");
        type = cmd.get<int>("type");
        stride = cmd.get<int>("stride");
        dilation = cmd.get<unsigned int>("dilation");
        bound = cmd.get<float>("bound");
        MIN_SZ = cmd.get<int>("size");
    } else {
        cmd.printErrors();
        cout << "Example: compute_flow P01_01 img_%07d.jpg flow_%s_%07d.jpg [options]" << endl;
        cmd.printMessage();
        return -1;
    }
    int gpuCounts = getCudaEnabledDeviceCount();
    cout << "Number of GPUs present " << gpuCounts << endl;

    setDevice(gpuID);
    printShortCudaDeviceInfo(getDevice());

    const Ptr<BroxOpticalFlow> &alg_brox = BroxOpticalFlow::create(
            cmd.get<float>("brox-alpha"),
            cmd.get<int>("brox-gamma"),
            cmd.get<float>("brox-scale-factor"),
            cmd.get<int>("brox-inner-iterations"),
            cmd.get<int>("brox-outer-iterations"),
            cmd.get<int>("brox-solver-iterations")
    );
    const Ptr<OpticalFlowDual_TVL1> &alg_tvl1 = OpticalFlowDual_TVL1::create(
            cmd.get<float>("tvl1-tau"),
            cmd.get<float>("tvl1-lambda"),
            cmd.get<float>("tvl1-theta"),
            cmd.get<int>("tvl1-nscales"),
            cmd.get<int>("tvl1-warps"),
            cmd.get<float>("tvl1-epsilon"),
            cmd.get<int>("tvl1-iterations"),
            cmd.get<float>("tvl1-scale-step"),
            cmd.get<float>("tvl1-gamma"),
            cmd.get<bool>("tvl1-use-initial-flow")
    );

    switch (type) {
        case 0:
            std::cout << "Brox parameters" << std::endl;
            std::cout << "- alpha: " << alg_brox->getFlowSmoothness() << std::endl;
            std::cout << "- gamma: " << alg_brox->getGradientConstancyImportance() << std::endl;
            std::cout << "- scale_factor: " << alg_brox->getPyramidScaleFactor() << std::endl;
            std::cout << "- inner_iterations: " << alg_brox->getInnerIterations() << std::endl;
            std::cout << "- outer_iterations: " << alg_brox->getOuterIterations() << std::endl;
            std::cout << "- solver_iterations: " << alg_brox->getSolverIterations() << std::endl;
            break;
        case 1:
            std::cout << "TVL1 parameters" << std::endl;
            std::cout << "- epsilon: " << alg_tvl1->getEpsilon() << std::endl;
            std::cout << "- lambda: " << alg_tvl1->getLambda() << std::endl;
            std::cout << "- tau: " << alg_tvl1->getTau() << std::endl;
            std::cout << "- theta: " << alg_tvl1->getTheta() << std::endl;
            std::cout << "- iterations: " << alg_tvl1->getNumIterations() << std::endl;
            std::cout << "- nscales: " << alg_tvl1->getNumScales() << std::endl;
            std::cout << "- warps: " << alg_tvl1->getNumWarps() << std::endl;
            std::cout << "- use_initial_flow: " << alg_tvl1->getUseInitialFlow() << std::endl;
            break;
    }


    std::queue<Mat> frame_queue;

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

        resized_frame = Mat(Size(width, height), CV_8UC3);

        resize(current_frame, resized_frame, Size(width, height), 0, 0, INTER_CUBIC);

        frame_queue.push(resized_frame.clone());
    }
    Size frame_size = Size(width, height);
    DEBUG_MSG("Queue filled");
    assert(frame_queue.size() == ((size_t) dilation));
    // len(frame_queue) == dilation

    unsigned int flow_index = 0;  // this is represents the position at which we *can* compute a flow field, although we don't
                         // always compute a flow field if temporal downsampling via a stride is specified.

    Mat frame0_rgb, frame1_rgb;
    Mat frame0 = Mat(frame_size, CV_8UC1);
    Mat frame1 = Mat(frame_size, CV_8UC1);
    Mat frame0_32 = Mat(frame_size, CV_32FC1);
    Mat frame1_32 = Mat(frame_size, CV_32FC1);

    GpuMat frame0GPU(frame_size, CV_32FC1);
    GpuMat frame1GPU(frame_size, CV_8UC1);
    GpuMat flowGPU(frame_size, CV_8UC1);
    Mat flowCPU(frame_size, CV_32FC2);
    Mat flow_channels[2];
    Mat img_u(frame_size, CV_8UC1);
    Mat img_v(frame_size, CV_8UC1);

    cap >> current_frame;
    frame_index++;

    while (!current_frame.empty()){
        DEBUG_MSG("Waiting on cap");
        DEBUG_MSG("Got frame " << frame_index);
        resized_frame = Mat(Size(width, height), CV_8UC3);
        resize(current_frame, resized_frame, frame_size, 0, 0, INTER_CUBIC);

        DEBUG_MSG("Push frame " << frame_index);
        frame_queue.push(resized_frame.clone());

        DEBUG_MSG("Pop frame " << frame_index - dilation);
        //extract frame 0 from the front of the queue and pop
        frame0_rgb = frame_queue.front().clone();
        frame_queue.pop();
        DEBUG_MSG("Popped frame " << frame_index - dilation);

        if (flow_index % stride == 0) {
            // Convert the image to grey and float
            cvtColor(frame0_rgb, frame0, COLOR_BGR2GRAY);
            //frame0.convertTo(frame0_32, CV_32FC1, 1.0 / 255.0, 0);

            cvtColor(resized_frame, frame1, COLOR_BGR2GRAY);
            //frame1.convertTo(frame1_32, CV_32FC1, 1.0 / 255.0, 0);


            DEBUG_MSG("Started frame upload");
            frame1GPU.upload(frame1);
            frame0GPU.upload(frame0);
            DEBUG_MSG("Completed frame upload");
            DEBUG_MSG("Started flow computation");
            switch (type) {
                case 0:
                    alg_brox->calc(frame0GPU, frame1GPU, flowGPU);
                case 1:
                    alg_tvl1->calc(frame0GPU, frame1GPU, flowGPU);
            }
            DEBUG_MSG("Completed flow computation");
            flowGPU.download(flowCPU);
            DEBUG_MSG("Downloaded flow");

            split(flowCPU, flow_channels);


            convertFlowToImage(flow_channels[0], img_u, -bound, bound);
            convertFlowToImage(flow_channels[1], img_v, -bound, bound);

            char *filename_x = new char[255];
            char *filename_y = new char[255];
            sprintf(filename_x, flow_format.c_str(), "x", flow_frame_number);
            sprintf(filename_y, flow_format.c_str(), "y", flow_frame_number);
            imwrite(input_folder + "/" + string(filename_x), img_u);
            imwrite(input_folder + "/" + string(filename_y), img_v);
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


