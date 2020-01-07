//************************************************************************
// compute_flow.cpp
// Computes OpenCV GPU Brox et al. [1] and Zach et al. [2] TVL1 Optical Flow
// Dependencies: OpenCV and Qt5 for iterating (sub)directories
// Author: Christoph Feichtenhofer -> Forked by Antonino Furnari
// Institution: Graz University of Technology
// Email: feichtenhofer@tugraz
// Date: Nov. 2015
// [1] T. Brox, A. Bruhn, N. Papenberg, J. Weickert. High accuracy optical flow estimation based on a theory for warping. ECCV 2004.
// [2] C. Zach, T. Pock, H. Bischof: A duality based approach for realtime TV-L 1 optical flow. DAGM 2007.
//************************************************************************

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>
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
int dilation = 1;
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

void converFlowMat(Mat& flowIn, Mat& flowOut,float min_range_, float max_range_)
{
    float value = 0.0f;
    for(int i = 0; i < flowIn.rows; i++)
    {
        float* Di = flowIn.ptr<float>(i);
        char* Ii = flowOut.ptr<char>(i);
        for(int j = 0; j < flowIn.cols; j++)
        {
            value = (Di[j]-min_range_)/(max_range_-min_range_);

            value *= 255;
            value = cvRound(value);

            Ii[j] = (char) value;
        }
    }
}

static void convertFlowToImage(const Mat &flowIn, Mat &flowOut,
        float lowerBound, float higherBound) {
    #define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flowIn.rows; ++i) {
        for (int j = 0; j < flowIn.cols; ++j) {
            float x = flowIn.at<float>(i,j);
            flowOut.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
        }
    }
    #undef CAST
}

int matMean(cv:: Mat img) {
    int sum = 0;
    int count = 0;
    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            sum+=img.at<uchar>(c,r);
            count++;
        }
    }
    int mean = sum/count;
    return mean;
}

int main( int argc, char *argv[] )
{
    GpuMat frame0GPU, frame1GPU, uGPU, vGPU;
    Mat currentFrame, resizedFrame;
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
        cout << "Usage: compute_flow input_folder img_formati flow_format [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printParams();
        cout << "Example: compute_flow P01_01 img_%07d.jpg flow_%s_%07d.jpg [options]" << endl;
        return 0;
    }

    if (argc > 2) {
        input_folder=argv[1];
        img_format=argv[2];
        flow_format=argv[3];

        gpuID = cmd.get<int>("gpuID");
        type = cmd.get<int>("type");
        stride = cmd.get<int>("stride");
        dilation = cmd.get<int>("dilation");
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

    std::cout << "Processing folder " << input_folder << std::endl;

    std::queue<cv::Mat> framequeue;
    std::queue<int> frame_index_queue;

    VideoCapture cap; //open video
    try {
        cap.open(input_folder+'/'+img_format);
    } catch (std::exception& e) {
        std::cout << e.what() << '\n'; //print exception
    }

    int nframes = 0, width = 0, height = 0;

    if( cap.isOpened() == 0 ) {
        cout << "Problem opening input file"<<endl;
        return -1; //exit with a -1 error
    }

    int frame_to_write=1;
    //NOW THE VIDEO IS OPEN AND WE CAN START GRABBING FRAMES

    //fill the frame queue
    int frame_index = -1;
    for (int ii=0; ii<dilation; ii++) {
        cap >> currentFrame;
        frame_index++;

        float factor = std::max<float>(MIN_SZ/currentFrame.cols, MIN_SZ/currentFrame.rows);

        width = std::floor(currentFrame.cols*factor);
        width -= width%2;
        height = std::floor(currentFrame.rows*factor);
        height -= height%2;

        resizedFrame = cv::Mat(Size(width,height),CV_8UC3);

        cv::resize(currentFrame,resizedFrame,cv::Size(width,height),0,0,INTER_CUBIC);

        framequeue.push(resizedFrame.clone());
        frame_index_queue.push(frame_index);
    }

    // cout << "Queue full" << endl;

    while (!currentFrame.empty()){
        resizedFrame = cv::Mat(Size(width,height),CV_8UC3);
        cv::resize(currentFrame,resizedFrame,cv::Size(width,height),0,0,INTER_CUBIC);

        framequeue.push(resizedFrame.clone());
        frame_index_queue.push(frame_index);

        //extract frame 0 from the front of the queue and pop
        frame0_rgb = framequeue.front().clone();
        framequeue.pop();
        int frame0_index = frame_index_queue.front();
        frame_index_queue.pop();

        // Allocate memory for the images
        //frame0_rgb = cv::Mat(Size(width,height),CV_8UC3);
        flow_rgb = cv::Mat(Size(width,height),CV_8UC3);
        motion_flow = cv::Mat(Size(width,height),CV_8UC3);
        frame0 = cv::Mat(Size(width,height),CV_8UC1);
        frame1 = cv::Mat(Size(width,height),CV_8UC1);
        frame0_32 = cv::Mat(Size(width,height),CV_32FC1);
        frame1_32 = cv::Mat(Size(width,height),CV_32FC1);

        // Convert the image to grey and float
        cvtColor(resizedFrame,frame1,CV_BGR2GRAY);
        frame1.convertTo(frame1_32,CV_32FC1,1.0/255.0,0);

        cvtColor(frame0_rgb,frame0,CV_BGR2GRAY);
        frame0.convertTo(frame0_32,CV_32FC1,1.0/255.0,0);

        DEBUG_MSG("(" << frame0_index  << ", " << frame_index << ")");
        switch(type){
            case 0:
                frame1GPU.upload(frame1_32);
                frame0GPU.upload(frame0_32);
                dflow(frame0GPU,frame1GPU,uGPU,vGPU);
            case 1:
                frame1GPU.upload(frame1);
                frame0GPU.upload(frame0);
                alg_tvl1(frame0GPU,frame1GPU,uGPU,vGPU);
        }

        uGPU.download(imgU);
        vGPU.download(imgV);

        //cv::resize(imgU,imgU,cv::Size(width_out,height_out),0,0,INTER_CUBIC);
        //cv::resize(imgV,imgV,cv::Size(width_out,height_out),0,0,INTER_CUBIC);

        float min_u_f = -bound;
        float max_u_f = bound;

        float min_v_f = -bound;
        float max_v_f = bound;

        cv::Mat img_u(imgU.rows, imgU.cols, CV_8UC1);
        cv::Mat img_v(imgV.rows, imgV.cols, CV_8UC1);

        convertFlowToImage(imgU, img_u, min_u_f, max_u_f);
        convertFlowToImage(imgV, img_v, min_v_f, max_v_f);

        //if(!fout.isOpened()) {
        //    fout.open(out_video,CV_FOURCC('M','P','4','2'),cap.get(CV_CAP_PROP_FPS),cv::Size(3*width,height),true);
        //}

        //if(!fout.isOpened()) {
        //    cout << "Problem opening output file"<<endl;
        //    return -1;
        //}

        //cv::resize(frame1_rgb,rgb_out,cv::Size(width_out,height_out),0,0,INTER_CUBIC);

        //Mat comb, comb2, img_uc, img_vc;

        //cvtColor(img_u,img_uc,CV_GRAY2BGR);
        //cvtColor(img_v,img_vc,CV_GRAY2BGR);

        //horizontally stack flow, then rgb
        //hconcat(img_uc,img_vc,comb);
        //hconcat(resizedFrame,comb,comb2);

        char *filename_x = new char[255];
        char *filename_y = new char[255];
        sprintf(filename_x,flow_format.c_str(),"x",frame_to_write);
        sprintf(filename_y,flow_format.c_str(),"y",frame_to_write);
        cv::imwrite(input_folder + "/" + string(filename_x),img_u);
        cv::imwrite(input_folder + "/" + string(filename_y),img_v);

        frame_to_write++;

        //fout.write(comb2);
#ifndef DEBUG
        cout << "." << flush;
#endif

        //frame1_rgb.copyTo(frame0_rgb);
        //cvtColor(frame0_rgb,frame0,CV_BGR2GRAY);
        //frame0.convertTo(frame0_32,CV_32FC1,1.0/255.0,0);

        //skip exactly frameSkip frames
        nframes++;
        for (int iskip = 0; iskip<stride-1; iskip++) {
            cap >> currentFrame;
            frame_index++;
        }
        cap >> currentFrame;
        frame_index++;
    }

    cout << endl;

    return 0;
}


