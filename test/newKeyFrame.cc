#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "CameraDevice.h"
#include "ImageFrame.h"
#include "VisionTracker.h"
#include "Viewer.h"
#include "Timer.h"

using namespace std;

int main(int argc, char** argv)
{

    // camera device 
    CameraIntrinsic K1(argv[1]);
    CameraDevice camera1(K1);
    printf("Camera1: %f %f %f %f %f %f %d %d\n", camera1.K.cx , camera1.K.cy ,camera1.K.fx ,camera1.K.fy ,
            camera1.K.k1 , camera1.K.k2, camera1.K.width ,camera1.K.height ) ;   

    bool isDataset = false;
    bool isVideo = false;
    cv::Mat Frame;
    if (!camera1.openDataset(argv[2])) {
        printf("Open dataset failed!\nTry open video file\n");

        if (!camera1.openVideo(argv[2])) {
            printf("Open video failed!\nTry open camera\n");

            if (!camera1.openCamera(atoi(argv[2]))) {
                printf("Open camera failed!\n");
                exit(0);
            }
        } else {
            isVideo = true;
        }
    } else {
        isDataset = true;
    }
    
    Mapping map(&K1);
    VisionTracker tracker(&K1, &map);                     // Vision Tracker

    Viewer viewer(&map, &tracker);
    std::thread* ptViewer = new std::thread(&Viewer::run, &viewer);
    //viewer.init();

    bool started = false;
    char cmd = ' ';
    cv::namedWindow("result");

    while (true) {


        if ( !camera1.getFrame(Frame, camera1.BGR) ) {
            printf("Get Frame failed!\n");
            break;
        }

        // If is dataset, control at each image
        if (isDataset || isVideo) {
            cmd = cv::waitKey(-1);
        }

        if (cmd == 's') {
            tracker.reset();
            started = true;
        }

        if (started) {

            ImageFrame newImgFrame(Frame, &K1);

            TIME_BEGIN()
            tracker.TrackMonocularNewKeyFrame(newImgFrame);
            TIME_END("One Frame")

            //viewer.requestDraw();

            if (newImgFrame.isKeyFrame) {

                printf("Draw KeyFrame...\n");

                for (int i = 0, _end = (int)newImgFrame.points.size(); i < _end; i++) {
                    cv::circle(Frame, newImgFrame.points[i], 2, cv::Scalar(0,0,255));
                    if (newImgFrame.ref[i] > 0) {
                        // draw fuse ref
                        cv::line(Frame, newImgFrame.points[i], newImgFrame.trackedPoints[newImgFrame.ref[i]], cv::Scalar(255,0,0));
                        // draw match fast
                        cv::line(Frame, newImgFrame.points[i], newImgFrame.mRefFrame->points[newImgFrame.ref[i]], cv::Scalar(0,255,0));
                    }
                }

            } else {
                
                printf("Draw internal Frame...\n");

                for (int i = 0, _end = (int)newImgFrame.trackedPoints.size(); i < _end; i++) { // draw result
                    if (newImgFrame.trackedPoints[i].x > 0) {
                        if (tracker.state != tracker.INITIALIZED) {
                            cv::line(Frame, newImgFrame.mRefFrame->points[i], 
                                    newImgFrame.trackedPoints[i],
                                    cv::Scalar(255, 0, 0));
                        } else {
                            cv::line(Frame, newImgFrame.mRefFrame->points[i], 
                                    newImgFrame.trackedPoints[i],
                                    cv::Scalar(0, 255, 0));
                        }
                    }
                    cv::circle(Frame, newImgFrame.mRefFrame->points[i], 
                            2,
                            cv::Scalar(0, 0, 0));
                }
            }
        }

        cv::imshow("result",Frame);
        cmd = cv::waitKey(33);
    }

    return 0;
}
