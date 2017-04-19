#ifndef IMAGEFRAME_H
#define IMAGEFRAME_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <mutex>

#include <boost/bimap.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <TooN/se3.h>
#include <TooN/so3.h>

//#include <cvd/image_io.h>
//#include <cvd/vision.h>
//#include <cvd/esm.h>

#include "CameraIntrinsic.h"

using namespace std;

typedef boost::bimap< int, cv::Point3f* > Map_2d_3d;
typedef Map_2d_3d::value_type Map_2d_3d_key_val;

class ImageFrame
{
    public:
        cv::Mat image;                         // image data
        //CVD::Image<unsigned char> SBI;         // small blurry image
        vector< cv::KeyPoint > keyPoints;      // original fast keypoints
        vector< cv::Point2f > points;          // Just point2f
        vector< cv::Point2f > undisPoints;     // Undistorted keypoints

        vector< cv::Point2f > trackedPoints;
        vector< cv::Point2f > undisTrackedPoints;

        cv::Mat descriptors;

        Map_2d_3d map_2d_3d;

        TooN::SE3<> mTcw;                      // Transformation from w to camera
        cv::Mat R, t;                          // Tcw R, t
        ImageFrame* mRefFrame;                 // Reference Frame
        CameraIntrinsic* K;                    // CameraIntrinsic

        bool isKeyFrame;
        vector< int > ref;                     // points reference to last keyFrame
                                               // only used for key frame

        ImageFrame() = default;
        explicit ImageFrame(const cv::Mat& frame, CameraIntrinsic* _K);
        explicit ImageFrame(const ImageFrame& imgFrame);

        void extractFAST(int lowNum = 400, int highNum = 500);
        void extractPatch();
        cv::Mat extractTrackedPatch();
        void computePatchDescriptorAtPoint(
                const cv::Point2f &pt, 
                const cv::Mat &im,
                float* desc);
        void trackPatchFAST(ImageFrame& refFrame);
        int opticalFlowFAST(ImageFrame& refFrame);
        vector< int > fuseFAST();
        void opticalFlowTrackedFAST(ImageFrame& lastFrame);
        //void SBITrackFAST(ImageFrame& refFrame);

        cv::Mat GetTwcMat();
        cv::Mat GetTcwMat();

};

#endif
