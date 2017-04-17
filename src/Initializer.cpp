#include "Initializer.h"

void Initializer::SetFirstFrame(ImageFrame *f)
{
    firstFrame = *f;
    lastFrame = firstFrame;
}

bool Initializer::TryInitialize(ImageFrame *f)
{
    // Track FAST Points
    //int num_tracked = f.opticalFlowFAST(*refFrame);
    // check trakced points ratio
    //double ratio_tracked = 
    //        num_tracked / (double)f.undisTrackedPoints.size();

    
    // optical flow fast features from last frame
    f->opticalFlowTrackedFAST(lastFrame);
    f->mRefFrame = &firstFrame;

    // essential matrix validation from first frame
    vector< int > pt_idx;
    vector< cv::Point2f > pt_1, pt_2;
    pt_idx.reserve(firstFrame.points.size());
    pt_1.reserve(firstFrame.points.size());
    pt_2.reserve(firstFrame.points.size());

    for (int i = 0, _end = (int)f->trackedPoints.size(); i < _end; i++) {
        if (f->trackedPoints[i].x > 0) {
            pt_1.push_back(firstFrame.undisPoints[i]);
            pt_2.push_back(f->undisTrackedPoints[i]);
            pt_idx.push_back(i);
        }
    }
    // essential matrix estimation validation
    cv::Mat inlier;
    TIME_BEGIN()
    cv::findEssentialMat(pt_1, pt_2, 
            (f->K->fx + f->K->fy)/2, cv::Point2d(f->K->cx, f->K->cy),
            cv::RANSAC, 0.9999, 2, inlier);
    TIME_END("TryInitialize: essential matrix estimation")
    int num_inliers = 0;
    for (int i = 0, _end = (int)pt_1.size(); i < _end; i++) {
        if (inlier.at< unsigned char >(i) == 0) {
            int idx = pt_idx[i];
            f->trackedPoints[idx].x = f->trackedPoints[idx].y = 0;
            f->undisTrackedPoints[idx].x = f->undisTrackedPoints[idx].y = 0;
        } else {
            num_inliers++;
        }
    }
    // upate last frame
    lastFrame = *f;
    
    double ratio_tracked = (double)num_inliers / (double)firstFrame.points.size();
    if (ratio_tracked < 0.5) {
        printf("Initialize Tracked points num too small!"
                " less than 0.3\n");
        return false;
    }

    // If tracked points less than 0.3, reset first Frame
    if (ratio_tracked < 0.3) {
        printf("Initialize Tracked points num too small!"
                " less than 0.3, Reset first frame.\n");
        f->extractFAST();
        f->trackedPoints = f->points;
        f->undisTrackedPoints = f->undisPoints;
        SetFirstFrame(f);
        return false;
    }

    return RobustTrackPose2D2D(firstFrame, *f);
}

bool Initializer::TryInitializeByG2O(ImageFrame *f)
{
    
}

bool Initializer::RobustTrackPose2D2D(ImageFrame &lf, ImageFrame &rf)
{
    // Track Pose 2D-2D
    // prepare tracked points
    std::vector< cv::Point2f > lp, rp;
    std::vector< int > pt_idx;
    lp.reserve(rf.undisTrackedPoints.size());
    rp.reserve(rf.undisTrackedPoints.size());
    pt_idx.reserve(rf.undisTrackedPoints.size());
    for (int i = 0, _end = (int)rf.undisTrackedPoints.size(); 
            i < _end; i++ ) {
        if (rf.undisTrackedPoints[i].x > 0) {
            lp.push_back(lf.undisPoints[i]);
            rp.push_back(rf.undisTrackedPoints[i]);
            pt_idx.push_back(i);
        }
    }

    //// check disparty
    //double disparty = 0;
    //for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
    //    disparty = disparty + (lp[i].x - rp[i].x)*(lp[i].x - rp[i].x)
    //            + (lp[i].y - rp[i].y)*(lp[i].y - rp[i].y);
    //}
    //disparty = sqrt(disparty/(double)lp.size()) ;
    //if ( disparty < K->width/32.0 ) {
    //    printf("Initialize disparty too small, less than %f average!\n", 
    //          K->width/32.0);
    //    return;
    //}

    // find essentialmat
    cv::Mat inliers;
    cv::Mat essential_matrix = cv::findEssentialMat(
            lp, rp, 
            (lf.K->fx + lf.K->fy)/2, cv::Point2d(lf.K->cx, lf.K->cy), 
            cv::RANSAC, 0.999, 2.0, inliers);
    int num_inliers = 0;
    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
        if (inliers.at<unsigned char>(i) == 1) {
            ++num_inliers;
        } else {
            int idx = pt_idx[i];
            rf.trackedPoints[idx].x = rf.trackedPoints[idx].y = 0;
            rf.undisTrackedPoints[idx].x = rf.undisTrackedPoints[idx].y = 0;
        }
    }
    double ratio_inliers = (double)num_inliers / (int)lp.size();
    if (ratio_inliers < 0.9) {
       printf("Initialize essential matrix inliers num too small!"
               " less than 0.9\n"); 
       return false;
    }
    cout << "essential_matrix: " << endl
        << essential_matrix << endl;

    // recovery pose
    cv::Mat R, t;
    cv::recoverPose(essential_matrix, lp, rp, R, t, 
            (lf.K->fx + lf.K->fy)/2, cv::Point2d(lf.K->cx, lf.K->cy), inliers);

    // triangulate points
    cv::Mat T1, T2, pts_4d;
    cv::hconcat(cv::Mat::eye(3,3,CV_64FC1), cv::Mat::zeros(3, 1, CV_64FC1), T1);
    cv::hconcat(R.t(), -t, T2);

    std::vector< cv::Point2f > pts_1, pts_2;
    std::vector< int > pt_idx_tri;
    pt_idx_tri.reserve(lf.undisPoints.size());
    pts_2.reserve(lf.undisPoints.size());
    pts_2.reserve(lf.undisPoints.size());

    for (int i = 0, _end = (int)lf.undisPoints.size(); i < _end; i++) {
        if ( rf.undisTrackedPoints[i].x > 0 ) {
            pt_idx_tri.push_back(i);
            pts_1.push_back(lf.K->pixel2device(
                        lf.undisPoints[i].x,
                        lf.undisPoints[i].y));
            pts_2.push_back(lf.K->pixel2device(
                        rf.undisTrackedPoints[i].x,
                        rf.undisTrackedPoints[i].y));
        }
    }
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    // check points 
    if( !CheckPoints(R.t(), -t, pts_4d) ) {
        return false;
    }

    cout << "R: " << endl
        << R << endl;
    cout << "t: " << endl
        << t << endl;

    cv::Mat Rt = R.t();
    cv::Mat _t = -t;

    mR = Rt.clone();
    mt = _t.clone();

    f.R = mR.clone();
    f.t = mt.clone();
}

bool Initializer::CheckPoints(cv::Mat R, cv::Mat t, cv::Mat &pts)
{
    // check parallax
    for (int i = 0; i < pts.cols; i++) {
        cv::Mat pts_3d(3, 1, CV_64FC1);
        double w = pts[i].at<double>(3);
        pts_3d.at<double>(0) = pts[i].at<double>(0) / w;
        pts_3d.at<double>(1) = pts[i].at<double>(1) / w;
        pts_3d.at<double>(2) = pts[i].at<double>(2) / w;

        //cv::Mat PO2 = 
    }
}
