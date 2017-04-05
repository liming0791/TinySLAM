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

    CameraIntrinsic K1(argv[1]);
    CameraDevice camera1(K1);
    printf("Camera1: %f %f %f %f %f %f %d %d\n", camera1.K.cx , camera1.K.cy ,camera1.K.fx ,camera1.K.fy ,
            camera1.K.k1 , camera1.K.k2, camera1.K.width ,camera1.K.height ) ;   

    bool isDataset = false;
    cv::Mat Frame;
    if (!camera1.openDataset(argv[2])) {
        printf("Open dataset failed!\nTry open camera\n");

        if (!camera1.openCamera(atoi(argv[2]))) {
            printf("Open camera failed!\n");
            exit(0);
        }
    } else {
        isDataset = true;
    }
    
    Mapping map(&K1);
    VisionTracker tracker(&K1, &map);                     // Vision Tracker

    //Viewer viewer(&map, &tracker);
    //std::thread* ptViewer = new std::thread(&Viewer::run, &viewer);

    ImageFrame *refImgFrame = NULL;

    bool isFirst = true;
    char cmd = ' ';
    cv::namedWindow("result");

    while (true) {
        if ( !camera1.getFrame(Frame, camera1.BGR) ) {
            printf("Get Frame failed!\n");
            break;
        }

        cout << "Got new Frame" << endl;

        //if (isDataset) {
        //    cmd = cv::waitKey(-1);
        //}

        if (cmd == 's') {
            if (isFirst) {              // fisrt 's'
                refImgFrame = new ImageFrame(Frame, &K1); 
                TIME_BEGIN()
                refImgFrame->extractFAST();
                refImgFrame->extractPatch();
                TIME_END("FAST")
                isFirst = false;
            } else {                                // second 's'
                if ( refImgFrame != NULL ) {
                    ImageFrame newImgFrame(Frame, &K1);
                    // optical flow result
                    newImgFrame.opticalFlowFAST(*refImgFrame);
                    tracker.TrackPose2D2D(*refImgFrame, newImgFrame);
                    map.InitMap(*refImgFrame, newImgFrame);
                    isFirst = true;
                    for (int i = 0, _end = (int)newImgFrame.trackedPoints.size(); i < _end; i++) { // draw result
                        if (newImgFrame.trackedPoints[i].x > 0) {
                            cv::line(Frame, refImgFrame->points[i], 
                                    newImgFrame.trackedPoints[i],
                                    cv::Scalar(0, 255, 0));
                        }
                    }
                }    
            }
        } else {                                
            if ( refImgFrame != NULL ) {
                cout << "Create new Frame" << endl;
                ImageFrame newImgFrame(Frame, &K1);
                // optical flow result
                cout << "Track Patch FAST" << endl;
                newImgFrame.trackPatchFAST(*refImgFrame);
                cout << "trackedPoints size: " << newImgFrame.trackedPoints.size() << endl;

                for (int i = 0, _end = (int)newImgFrame.trackedPoints.size(); i < _end; i++) { // draw result
                    if (newImgFrame.trackedPoints[i].x > 0) {
                        cv::line(Frame, refImgFrame->points[i], 
                                newImgFrame.trackedPoints[i],
                                cv::Scalar(0, 255, 0));
                    }
                }
            }    
        }

        cv::imshow("result",Frame);
        cmd = cv::waitKey(33);
    }

    //cv::waitKey();

    //viewer.requestFinish();

    return 0;
}
