#ifndef MAPPING_H
#define MAPPING_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <queue>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "ImageFrame.h"
#include "CameraIntrinsic.h"

using namespace std;

class VisionTracker;

class Mapping
{
    public:
        CameraIntrinsic* K;
        std::set< cv::Point3f* > mapPoints;
        std::vector< ImageFrame* > keyFrames;

        std::queue< ImageFrame > keyFrameQueue;

        Mapping() = default;
        Mapping(CameraIntrinsic* _K):K(_K){};
        void SetTracker(VisionTracker* _tracker);
        void InitMap(ImageFrame& lf, ImageFrame& rf);
        void Run();
        void InsertKeyFrame(ImageFrame &f);
        void TriangulateNewPoints(ImageFrame& lf, ImageFrame& rf);
        void AddFrameToQ(ImageFrame &f);
        bool CheckPoints(cv::Mat &R, 
                cv::Mat &t, 
                cv::Mat &pts);
        void ClearMap();

        std::mutex mMapMutex; 
        std::mutex mQueueMutex;

        VisionTracker* tracker;
};

#endif
