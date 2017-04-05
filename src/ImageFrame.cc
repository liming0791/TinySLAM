#include "ImageFrame.h"
#include "Timer.h"

ImageFrame::ImageFrame(const cv::Mat& frame, CameraIntrinsic* _K): SBI(CVD::ImageRef(32, 24)), R(cv::Mat::eye(3,3, CV_64FC1)), t(cv::Mat::zeros(3,1, CV_64FC1)), K(_K)
{
    if (frame.channels()==3) {
        cv::cvtColor(frame, image, CV_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, image, CV_BGRA2GRAY);
    } else {
        frame.copyTo(image);
    }

    cv::Mat s_img;
    cv::resize(image, s_img, cv::Size(32, 24));
    cv::blur(s_img, s_img, cv::Size(3,3));

    memcpy(SBI.begin(), s_img.data, 32*24*sizeof(unsigned char));
}

ImageFrame::ImageFrame(const ImageFrame& imgFrame): 
    image(imgFrame.image), SBI(imgFrame.SBI), keyPoints(imgFrame.keyPoints), 
    mTcw(imgFrame.mTcw), R(imgFrame.R), t(imgFrame.t), mRefFrame(imgFrame.mRefFrame), K(imgFrame.K)
{
    
}

void ImageFrame::extractFAST()
{
    int thres = 40;
    int low = 400;
    int high = 500;
    cv::FAST(image, keyPoints, thres);

    int iter = 0;
    while( iter < 20 && ( (int)keyPoints.size() < low || (int)keyPoints.size() > high ) )
    {
       if ((int)keyPoints.size() < low) {
            thres -= thres * 0.05;
            cv::FAST(image, keyPoints, thres);
       } else if ((int)keyPoints.size() > high) {
            thres += thres * 0.05;
            cv::FAST(image, keyPoints, thres);
       }
       iter++;
    }

    points.reserve(keyPoints.size());       // use reserve to improve performance
    points.resize(0);
    undisPoints.reserve(keyPoints.size());
    undisPoints.resize(0);
    for (int i = 0, _end = (int)keyPoints.size(); i < _end; i++) {
        points.push_back( keyPoints[i].pt );
        undisPoints.push_back( K->undistort(points[i].x, points[i].y) );
    }

    trackedPoints = points;
    undisTrackedPoints = undisPoints;
    
}

void ImageFrame::extractPatch()
{
    if (points.size() == 0) {
        return;
    }

    descriptors.create(points.size(), 49, CV_32FC1);
    for (int i = 0, _end = (int)points.size(); i < _end; i++) {
        computePatchDescriptorAtPoint(points[i], image,
                descriptors.ptr<float>(i));
    }
}

void ImageFrame::computePatchDescriptorAtPoint(const cv::Point2f &pt, 
        const cv::Mat &im, float* desc)
{
    const int PATCH_SIZE = 7;
    const int HALF_PATCH_SIZE = 3; 

    int centerIdx = pt.x + pt.y*im.cols;
    int startIdx = centerIdx - HALF_PATCH_SIZE*im.cols - HALF_PATCH_SIZE;

    const unsigned char* data = im.data;

    // Ave
    float aveVal = 0; 
    int nowIdx=startIdx;
    for (int r = 0; r < PATCH_SIZE; r++,nowIdx += im.cols) {
        for (int c = 0; c < PATCH_SIZE; c++,nowIdx++) {
            aveVal += float(data[nowIdx]);
        }
    }
    aveVal =  aveVal / (PATCH_SIZE*PATCH_SIZE);

    // Devi
    float devi = 0;
    nowIdx = startIdx;
    for (int r = 0; r < PATCH_SIZE; r++,nowIdx += im.cols) {
        for (int c = 0; c < PATCH_SIZE; c++,nowIdx++) {
            float val = float(data[nowIdx]);
            devi += (val - aveVal)*(val - aveVal);
        }
    }
    devi /= (PATCH_SIZE*PATCH_SIZE);
    devi = sqrt(devi);

    // Desc
    int desIdx = 0;
    nowIdx = startIdx;
    for (int r = 0; r < PATCH_SIZE; r++,nowIdx += im.cols) {
        for (int c = 0; c < PATCH_SIZE; c++,nowIdx++,desIdx++) {
            desc[desIdx] = (float(data[nowIdx]) - aveVal) / devi;
        }
    }
    
}

void ImageFrame::trackPatchFAST(ImageFrame& refFrame)
{

    mRefFrame = &refFrame; 

    // extract FAST
    TIME_BEGIN()
    extractFAST();
    TIME_END("FAST time: ")

    // extractPatch
    TIME_BEGIN()
    extractPatch();
    TIME_END("Extract Patch time: ")

    if ( descriptors.empty() || descriptors.rows == 0)
        return;

    // match
    cv::BFMatcher matcher(cv::NORM_L2, true);
    vector< cv::DMatch > matches;
    TIME_BEGIN()
    matcher.match(refFrame.descriptors, descriptors, matches);
    TIME_END("Match time: ")

    cout << "find matches: " << matches.size() << endl;

    trackedPoints.resize(refFrame.points.size());
    undisTrackedPoints.resize(refFrame.points.size());

    std::fill(trackedPoints.begin(), trackedPoints.end(), 
            cv::Point2f(0, 0));
    std::fill(undisTrackedPoints.begin(), undisTrackedPoints.end(), 
            cv::Point2f(0, 0));

    float maxDis = 0, minDis = 999;
    for (int i = 0, _end = (int)matches.size(); i < _end; i++) {
       if (matches[i].distance > maxDis)
           maxDis = matches[i].distance;
       if (matches[i].distance < minDis)
           minDis = matches[i].distance;
    }
    float thres = minDis + (maxDis - minDis) * 0.2;
    
    for (int i = 0, _end = (int)matches.size(); i < _end; i++) {
        if (matches[i].distance < thres)
        {
            trackedPoints[matches[i].queryIdx] = points[matches[i].trainIdx];
            undisTrackedPoints[matches[i].queryIdx] = 
                K->undistort(points[matches[i].trainIdx].x,
                        points[matches[i].trainIdx].y);
        }
    }

    cout << " track patch FAST done." << endl;

    // homography estimation
    //cv::Mat inliner;
    //TIME_BEGIN()
    //    cv::findHomography(refFrame.undisPoints, undisTrackedPoints, 
    //        cv::RANSAC, 3, inliner);
    //TIME_END("Homography estimation: ")

    //for (int i = 0, _end = (int)trackedPoints.size(); i < _end; i++) {
    //    if (inliner.at<unsigned char>(i) == 1) {

    //    } else {
    //        trackedPoints[i].x = trackedPoints[i].y = 0;
    //        undisTrackedPoints[i].x = undisTrackedPoints[i].y = 0;
    //    }
    //}
}

void ImageFrame::opticalFlowFAST(ImageFrame& refFrame)
{
    mRefFrame = &refFrame;

    // optical flow
    cv::Mat status, err;
    TIME_BEGIN()
    cv::calcOpticalFlowPyrLK(refFrame.image, image, 
            refFrame.points, trackedPoints, status, err) ;
    TIME_END("Optical Flow")

    // homography estimation validation 
    undisTrackedPoints.reserve(trackedPoints.size());
    undisTrackedPoints.resize(0);
    for (int i = 0, _end = (int)trackedPoints.size(); i < _end; i++) {
        undisTrackedPoints.push_back( 
                K->undistort(trackedPoints[i].x, trackedPoints[i].y) );
    }

    cv::Mat inliner;
    TIME_BEGIN()
    cv::findHomography(refFrame.undisPoints, undisTrackedPoints, 
            cv::RANSAC, 3, inliner);
    TIME_END("Homography estimation")

    for (int i = 0, _end = (int)trackedPoints.size(); i < _end; i++) {
        if (inliner.at<unsigned char>(i) == 1) {

        } else {
            trackedPoints[i].x = trackedPoints[i].y = 0;
            undisTrackedPoints[i].x = undisTrackedPoints[i].y = 0;
        }
    }

}

void ImageFrame::opticalFlowTrackedFAST(ImageFrame& lastFrame)
{
    mRefFrame = &lastFrame;

    vector< cv::Point2f > pts, undis_pts, undis_flow_pts,flow_pts;
    vector< int > idxs;
    pts.reserve(lastFrame.trackedPoints.size());
    undis_pts.reserve(lastFrame.trackedPoints.size());
    flow_pts.reserve(lastFrame.trackedPoints.size());
    undis_flow_pts.reserve(lastFrame.trackedPoints.size());
    idxs.reserve(lastFrame.trackedPoints.size());

    for (int i = 0, _end = (int)lastFrame.trackedPoints.size(); 
            i < _end; i++) {
        if (lastFrame.trackedPoints[i].x > 0) {
            pts.push_back(lastFrame.trackedPoints[i]);
            undis_pts.push_back(lastFrame.undisTrackedPoints[i]);
            idxs.push_back(i);
        }
    }

    cv::Mat status, err;
    TIME_BEGIN()
        cv::calcOpticalFlowPyrLK(lastFrame.image, image, 
                pts, flow_pts, status, err) ;
    TIME_END("Optical Flow")

    for (int i = 0, _end = (int)flow_pts.size(); i < _end; i++) {
        undis_flow_pts.push_back( 
                K->undistort(flow_pts[i].x, flow_pts[i].y) );
    }

    cv::Mat inliner;
    TIME_BEGIN()
    cv::findHomography(undis_pts, undis_flow_pts, 
            cv::RANSAC, 3, inliner);
    TIME_END("Homography estimation")

    trackedPoints.resize(lastFrame.trackedPoints.size());
    undisTrackedPoints.resize(lastFrame.trackedPoints.size());
    fill(trackedPoints.begin(), 
            trackedPoints.end(), cv::Point2f(-1,-1));
    fill(undisTrackedPoints.begin(), 
            undisTrackedPoints.end(), cv::Point2f(-1,-1));
    for (int i = 0, _end = (int)undis_flow_pts.size(); i < _end; i++) {
        if (inliner.at<unsigned char>(i) == 1) {
          trackedPoints[idxs[i]] = flow_pts[i];
          undisTrackedPoints[idxs[i]] = undis_flow_pts[i];
        } else {

        }
    }

}

void ImageFrame::SBITrackFAST(ImageFrame& refFrame)
{
    mRefFrame = &refFrame;

    CVD::Homography<8> homography;
    CVD::StaticAppearance appearance;
    CVD::Image< TooN::Vector<2> > greImg 
            = CVD::Internal::gradient<TooN::Vector<2>, unsigned char>(refFrame.SBI);
    CVD::Internal::esm_opt(homography, appearance, refFrame.SBI, greImg, SBI, 40, 1e-8, 1.0);
    TooN::Matrix<3> H = homography.get_matrix();


    H(0,2) = H(0,2) * 20.f;
    H(1,2) = H(1,2) * 20.f;
    H(2,0) = H(2,0) / 20.f;
    H(2,1) = H(2,1) / 20.f;

    keyPoints.resize(0);

    for (int i = 0, _end = (int)refFrame.keyPoints.size(); i < _end; i++ ) {
        TooN::Vector<3> P;
        P[0] = refFrame.keyPoints[i].pt.x;
        P[1] = refFrame.keyPoints[i].pt.y;
        P[2] = 1;
        TooN::Vector<3> n_P = H * P;
        keyPoints.push_back(cv::KeyPoint(n_P[0]/n_P[2], n_P[1]/n_P[2], 10));
    }

}

cv::Mat ImageFrame::GetTcwMat()
{
    if (R.empty() || t.empty())
        return cv::Mat();

    cv::Mat res = cv::Mat::eye(4, 4, CV_64FC1);
    R.copyTo(res.rowRange(0,3).colRange(0,3));
    t.copyTo(res.rowRange(0,3).col(3));
    return res;
}

cv::Mat ImageFrame::GetTwcMat()
{
    if (R.empty() || t.empty() )
        return cv::Mat();

    cv::Mat res = cv::Mat::eye(4, 4, CV_64FC1);
    cv::Mat Rt = R.t();
    cv::Mat _t = -t;
    Rt.copyTo(res.rowRange(0,3).colRange(0,3));
    _t.copyTo(res.rowRange(0,3).col(3));
    return res;
}
