#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>


#include "ImageFrame.h"
#include "Timer.h"

using namespace std;

class Initializer
{
    public:
        ImageFrame firstFrame;
        ImageFrame lastFrame;
        vector< cv::Point3f > mapPoints;

        Initializer() = default;
        ~Initializer() = default;

        void SetFirstFrame(ImageFrame *f);
        bool TryInitialize(ImageFrame *f);
        bool TryInitializeByG2O(ImageFrame *f);
        bool RobustTrackPose2D2D(ImageFrame &lf, 
                ImageFrame &rf);
        bool CheckPoints(cv::Mat R, cv::Mat t, cv::Mat &pts);

};

#endif
