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

    cv::Mat Frame;
    if (!camera1.openCamera(atoi(argv[2]))) {
        printf("Open camera failed!\n");
        exit(0);
    }
    
    Mapping map(&K1);
    VisionTracker tracker(&K1, &map);                     // Vision Tracker

    Viewer viewer(&map, &tracker);
    std::thread* ptViewer = new std::thread(&Viewer::run, &viewer);

    bool started = false;
    char cmd = ' ';
    while (true) {

        TIME_BEGIN()

        if ( !camera1.getFrame(Frame, camera1.BGR) ) {
            printf("Get Frame failed!\n");
            break;
        }

        if (cmd == 's') {
            tracker.reset();
            started = true;
        }

        if (started) {
            ImageFrame newImgFrame(Frame, &K1);
            tracker.TrackMonocularDirect(newImgFrame);

            for (int i = 0, _end = (int)tracker.refFrame.points.size(); i < _end; i++) { // draw result
                if (tracker.state != tracker.INITIALIZED) {
                    if (newImgFrame.trackedPoints[i].x > 0) {
                        cv::line(Frame, tracker.refFrame.points[i], 
                            newImgFrame.trackedPoints[i],
                            cv::Scalar(255, 0, 0));
                    }
                } else {
                    //if (newImgFrame.trackedPoints[i].x > 0) {
                    //    cv::line(Frame, tracker.refFrame.points[i], 
                    //        newImgFrame.trackedPoints[i],
                    //        cv::Scalar(0, 255, 0));
                    //}
                }
            }
        }

        TIME_END("One Frame")

        cv::imshow("result",Frame);
        cmd = cv::waitKey(33);
    }

    return 0;
}
