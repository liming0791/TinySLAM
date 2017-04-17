#include "ImageFrame.h"
#include "Timer.h"
#include "Converter.h"

ImageFrame::ImageFrame(const cv::Mat& frame, CameraIntrinsic* _K): 
    SBI(CVD::ImageRef(32, 24)), R(cv::Mat::eye(3,3, CV_64FC1)), 
    t(cv::Mat::zeros(3,1, CV_64FC1)), mRefFrame(this), K(_K), isKeyFrame(false)
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
    points(imgFrame.points), undisPoints(imgFrame.undisPoints),
    trackedPoints(imgFrame.trackedPoints), 
    undisTrackedPoints(imgFrame.undisTrackedPoints),
    descriptors(imgFrame.descriptors), map_2d_3d(imgFrame.map_2d_3d),
    mTcw(imgFrame.mTcw), R(imgFrame.R), t(imgFrame.t), mRefFrame(imgFrame.mRefFrame), 
    K(imgFrame.K), isKeyFrame(imgFrame.isKeyFrame), ref(imgFrame.ref)
{
    
}

void ImageFrame::extractFAST(int lowNum, int highNum)
{
    int thres = 40;
    int low = lowNum;
    int high = highNum;
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

    // if trackedPoints and undisTrackedPoints are empty, fill them
    if (trackedPoints.empty() && undisTrackedPoints.empty()) {
        trackedPoints = points;
        undisTrackedPoints = undisPoints;
    }
    
}

vector< int > ImageFrame::fuseFAST()
{

    ref = vector< int > (points.size(), -1);

    int num_fuse = 0;
    for (int i = 0, _end = (int)trackedPoints.size(); i < _end; i++) {
        if (trackedPoints[i].x > 0) {
            bool findRef = false;
            for (int j = 0, _end = (int)points.size(); j < _end; j++) {
                double dis = (trackedPoints[i].x - points[j].x)*
                        (trackedPoints[i].x - points[j].x) +
                        (trackedPoints[i].y - points[j].y)*
                        (trackedPoints[i].y - points[j].y);
                if (dis < 4) {
                    ref[j] = i;
                    findRef = true;
                    num_fuse++;
                    break;
                }
            }
            
            //if (!findRef) {
            //    points.push_back(trackedPoints[i]);
            //    undisPoints.push_back(
            //            K->undistort(trackedPoints[i].x, trackedPoints[i].y));
            //    ref.push_back(i);
            //}
        }
    }

    printf("Fuse FAST num: %d,  ratio: %f\n", num_fuse, 
            (double)num_fuse / (double)points.size());

    //trackedPoints = points;
    //undisTrackedPoints = undisPoints;

    return ref;
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

cv::Mat ImageFrame::extractTrackedPatch()
{
    if (trackedPoints.size() == 0) {
        return cv::Mat();
    }

    cv::Mat res(trackedPoints.size(), 49, CV_32FC1);
    for (int i = 0, _end = (int)trackedPoints.size(); i < _end; i++) {
        computePatchDescriptorAtPoint(trackedPoints[i], image,
                res.ptr<float>(i));
    }

    return res;
}

void ImageFrame::computePatchDescriptorAtPoint(const cv::Point2f &pt, 
        const cv::Mat &im, float* desc)
{
    const int PATCH_SIZE = 7;
    const int HALF_PATCH_SIZE = 3; 

    if (pt.x < HALF_PATCH_SIZE || pt.x > im.cols - HALF_PATCH_SIZE
            || pt.y < HALF_PATCH_SIZE || pt.y > im.rows - HALF_PATCH_SIZE) {
        return;
    }

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
}

int ImageFrame::opticalFlowFAST(ImageFrame& refFrame)
{
    // set reference frame
    mRefFrame = &refFrame;

    // optical flow points
    cv::Mat status, err;
    TIME_BEGIN()
    cv::calcOpticalFlowPyrLK(refFrame.image, image, 
            refFrame.points, trackedPoints, status, err) ;
    TIME_END("Optical Flow")

    // undistort trackedPoints
    undisTrackedPoints.reserve(trackedPoints.size());
    undisTrackedPoints.resize(0);
    for (int i = 0, _end = (int)trackedPoints.size(); i < _end; i++) {
        undisTrackedPoints.push_back( 
                K->undistort(trackedPoints[i].x, trackedPoints[i].y) );
    }

    // prepare points
    vector< cv::Point2f > e_pt1, e_pt2;
    vector< int > e_idx;
    e_pt1.reserve(undisTrackedPoints.size());
    e_pt2.reserve(undisTrackedPoints.size());
    e_idx.reserve(undisTrackedPoints.size());

    // estimation validation by **Patch** and **disparty**
    TIME_BEGIN()
    cv::Mat desc = extractTrackedPatch();
    for (int i = 0, _end = (int)undisTrackedPoints.size(); i < _end; i++) {
        double dist = cv::norm(
                refFrame.descriptors.row(i),
                desc.row(i), cv::NORM_L2);
        //double disparty = cv::norm(
        //        refFrame.points[i] - trackedPoints[i]);
        //cout << "Patch dist : " << dist << endl;
        //cout << "Disparty: " << disparty << endl;
        if (dist <= 50 && status.at<unsigned char>(i) == 1) {
            e_pt1.push_back(refFrame.undisPoints[i]);
            e_pt2.push_back(undisTrackedPoints[i]);
            e_idx.push_back(i);
        } else {
            trackedPoints[i].x = trackedPoints[i].y = 0;
            undisTrackedPoints[i].x = undisTrackedPoints[i].y = 0;
        }
    }
    TIME_END("patch estimation validation")

    // inlier estimation
    cv::Mat inlier;
    // essential matrix estimation validation
    TIME_BEGIN()
    cv::findEssentialMat(e_pt1, e_pt2, 
            (K->fx + K->fy)/2, cv::Point2d(K->cx, K->cy),
            cv::RANSAC, 0.9999, 2, inlier);
    TIME_END("essential matrix estimation")

    int num_tracked = 0;
    for (int i = 0, _end = (int)e_idx.size(); i < _end; i++) {
        if (inlier.at<unsigned char>(i) == 1) {
            ++num_tracked;
        } else {
            trackedPoints[e_idx[i]].x = trackedPoints[e_idx[i]].y = 0;
            undisTrackedPoints[e_idx[i]].x = undisTrackedPoints[e_idx[i]].y = 0;
        }
    }

    return num_tracked;
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

    // essential matrix estimation validation
    //cv::Mat inlier;
    //TIME_BEGIN()
    //cv::findEssentialMat(undis_pts, undis_flow_pts, 
    //        (K->fx + K->fy)/2, cv::Point2d(K->cx, K->cy),
    //        cv::RANSAC, 0.9999, 2, inlier);
    //TIME_END("essential matrix estimation")

    trackedPoints.resize(lastFrame.trackedPoints.size());
    undisTrackedPoints.resize(lastFrame.trackedPoints.size());
    fill(trackedPoints.begin(), 
            trackedPoints.end(), cv::Point2f(0,0));
    fill(undisTrackedPoints.begin(), 
            undisTrackedPoints.end(), cv::Point2f(0,0));
    for (int i = 0, _end = (int)undis_flow_pts.size(); i < _end; i++) {
        //if (inlier.at<unsigned char>(i) == 1) {
          trackedPoints[idxs[i]] = flow_pts[i];
          undisTrackedPoints[idxs[i]] = undis_flow_pts[i];
        //} else {

        //}
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

cv::Mat ImageFrame::GetTwcMat()
{
    if (R.empty() || t.empty())
        return cv::Mat();

    cv::Mat res = cv::Mat::eye(4, 4, CV_64FC1);
    R.copyTo(res.rowRange(0,3).colRange(0,3));
    t.copyTo(res.rowRange(0,3).col(3));
    return res;
}

cv::Mat ImageFrame::GetTcwMat()
{
    if (R.empty() || t.empty() )
        return cv::Mat();

    cv::Mat res = cv::Mat::eye(4, 4, CV_64FC1);
    cv::Mat Rt = R.t();
    cv::Mat _t = -Rt*t;
    Rt.copyTo(res.rowRange(0,3).colRange(0,3));
    _t.copyTo(res.rowRange(0,3).col(3));
    return res;
}
