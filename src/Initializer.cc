#include "Initializer.h"

void Initializer::SetFirstFrame(ImageFrame *f)
{
    f->extractFAST();
    f->extractPatch();
    firstFrame = *f;
    lastFrame = firstFrame;
}

bool Initializer::TrackFeatureAndCheck(ImageFrame *f)
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

    if (ratio_tracked < 0.5) {
        printf("Initialize Tracked points num too small!"
                " less than 0.5\n");
        return false;
    }

    return true;
}

bool Initializer::TryInitialize(ImageFrame *f)
{
    if (!TrackFeatureAndCheck(f)) {
        return false;
    }    
    return RobustTrackPose2D2D(firstFrame, *f);
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
    double disparty = 0;
    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
        disparty = disparty + (lp[i].x - rp[i].x)*(lp[i].x - rp[i].x)
                + (lp[i].y - rp[i].y)*(lp[i].y - rp[i].y);
    }
    disparty = sqrt(disparty/(double)lp.size()) ;
    if ( disparty < lf.K->width/32.0 ) {
        printf("Initialize disparty too small, less than %f average!\n", 
              lf.K->width/32.0);
        return false ;
    }

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
    cv::hconcat(R, t, T2);

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

    bool checkPassed = false;
    TIME_BEGIN()
    checkPassed = CheckPoints(R,t,pts_4d);
    TIME_END("check points")

    if( !checkPassed ) {
        return false;
    }

    // insert map points
    for (int i = 0; i < pts_4d.cols; i++) {
        float w = pts_4d.at<float>(3, i);
        if (w!=0) {
            cv::Point3f* mpt = new cv::Point3f(
                    pts_4d.at<float>(0, i)/w,
                    pts_4d.at<float>(1, i)/w,
                    pts_4d.at<float>(2, i)/w
                    );
            cout << *mpt << endl;
            map->mapPoints.insert(mpt);
            lf.map_2d_3d.insert(
                    Map_2d_3d_key_val(pt_idx_tri[i], mpt)
                    );
        }
    }

    // insert key frame
    ImageFrame* nkf = new ImageFrame(firstFrame); 
    map->keyFrames.push_back(nkf);
    resFirstFrame = nkf;

    cout << "R: " << endl
        << R << endl;
    cout << "t: " << endl
        << t << endl;

    rf.R = R.clone();
    rf.t = t.clone();

    return true;
}

bool Initializer::TryInitializeByG2O(ImageFrame *f)
{
    if (!TrackFeatureAndCheck(f)) {
        return false;
    }
    return RobustTrackPose2D2DG2O(firstFrame, *f);
}

bool Initializer::RobustTrackPose2D2DG2O(ImageFrame &lf, ImageFrame &rf)
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
    double disparty = 0;
    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
        disparty = disparty + (lp[i].x - rp[i].x)*(lp[i].x - rp[i].x)
                + (lp[i].y - rp[i].y)*(lp[i].y - rp[i].y);
    }
    disparty = sqrt(disparty/(double)lp.size()) ;
    if ( disparty < lf.K->width/32.0 ) {
        printf("Initialize disparty too small, less than %f average!\n", 
              lf.K->width/32.0);
        return false ;
    }

    //// find essentialmat
    //cv::Mat inliers;
    //cv::Mat essential_matrix = cv::findEssentialMat(
    //        lp, rp, 
    //        (lf.K->fx + lf.K->fy)/2, cv::Point2d(lf.K->cx, lf.K->cy), 
    //        cv::RANSAC, 0.999, 2.0, inliers);
    //int num_inliers = 0;
    //for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
    //    if (inliers.at<unsigned char>(i) == 1) {
    //        ++num_inliers;
    //    } else {
    //        int idx = pt_idx[i];
    //        rf.trackedPoints[idx].x = rf.trackedPoints[idx].y = 0;
    //        rf.undisTrackedPoints[idx].x = rf.undisTrackedPoints[idx].y = 0;
    //    }
    //}
    //double ratio_inliers = (double)num_inliers / (int)lp.size();
    //if (ratio_inliers < 0.9) {
    //   printf("Initialize essential matrix inliers num too small!"
    //           " less than 0.9\n"); 
    //   return false;
    //}
    //cout << "essential_matrix: " << endl
    //    << essential_matrix << endl;

    // recovery pose
    // Optimizer
    g2o::SparseOptimizer optimizer;
    
    // linear solver
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = 
        new g2o::LinearSolverCSparse< g2o::BlockSolver_6_3::PoseMatrixType >();

    // block solver
    g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3( linearSolver );

    // optimization algorithm
    g2o::OptimizationAlgorithmLevenberg* algorithm = 
        new g2o::OptimizationAlgorithmLevenberg( block_solver );

    optimizer.setAlgorithm( algorithm );
    optimizer.setVerbose( false );

    cout << "add pose vertices." << endl;
    // add pose vertices
    g2o::VertexSE3Expmap* v1 = new g2o::VertexSE3Expmap();
    v1->setId(0);
    v1->setFixed(true);
    v1->setEstimate(g2o::SE3Quat());
    optimizer.addVertex(v1);

    g2o::VertexSE3Expmap* v2 = new g2o::VertexSE3Expmap();
    v2->setId(1);
    v2->setEstimate(g2o::SE3Quat());
    optimizer.addVertex(v2);

    cout << "add 3d points vertices" << endl;
    // add 3d point vertices
    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId(2+i);
        double z = 10;
        double x = ( lp[i].x - lf.K->cx ) / lf.K->fx * z;
        double y = ( lp[i].y - lf.K->cy ) / lf.K->fy * z;
        v->setMarginalized(true);
        v->setEstimate( Eigen::Vector3d(x, y, z) );
        optimizer.addVertex( v );
    }

    cout << "add camera parameters" << endl;
    // prepare camera parameters
    g2o::CameraParameters* camera = 
        new g2o::CameraParameters( 
                (lf.K->fx + lf.K->fy)/2, 
                Eigen::Vector2d(lf.K->cx, lf.K->cy), 
                0 );
    camera->setId(0);
    optimizer.addParameter(camera);

    cout << "add edges" << endl;
    // prepare edges
    vector< g2o::EdgeProjectXYZ2UV* > edges;
    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, 
                dynamic_cast< g2o::VertexSBAPointXYZ* > (optimizer.vertex(i+2)) );
        edge->setVertex( 1, 
                dynamic_cast< g2o::VertexSE3Expmap* > (optimizer.vertex(0)) );

        edge->setMeasurement( Eigen::Vector2d(lp[i].x, lp[i].y) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);

        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back( edge );
    }

    for (int i = 0, _end = (int)rp.size(); i < _end; i++) {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast< g2o::VertexSBAPointXYZ* > (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap* > (optimizer.vertex(1)) );

        edge->setMeasurement( Eigen::Vector2d(rp[i].x, rp[i].y) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);

        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back( edge );
    }

    cout << "optimization" << endl;
    // optimization
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);

    // num of inliers
    vector< int > inliers(edges.size()/2, 1);
    int num_vert = (int)inliers.size();
    for ( int i = 0, _end = (int)edges.size(); i < _end; i++) {
        g2o::EdgeProjectXYZ2UV* e = edges[i];
        e->computeError();
        if (e->chi2() > 1) {
            inliers[i%num_vert] = 0;
            cout << "error = " << e->chi2() << endl;
        } 
    }
    int num_inliers = 0;
    for ( int i = 0, _end = (int)inliers.size(); i < _end; i++) {
        num_inliers += inliers[i];
    }
    cout << "num inliers: " << num_inliers << endl;
    // check inliers
    double ratio_inlier = (double)num_inliers / (double)lp.size();
    if (ratio_inlier < 0.8) {
        printf("Inliers too small, less than 0.8 !\n");
        return false;
    }

    // SE3 estimate
    g2o::VertexSE3Expmap* v = 
            dynamic_cast< g2o::VertexSE3Expmap* >( optimizer.vertex(1) );
    Eigen::Isometry3d pose = v->estimate();
    // set optimization pose result, do not forget this step
    Eigen::Matrix4d T = pose.matrix();
    cv::Mat res;
    cv::eigen2cv(T, res);
    cv::Mat R = res.rowRange(0,3).colRange(0,3);
    cv::Mat t = res.rowRange(0,3).col(3);

    // points estimate
    cv::Mat pts_4d(4, (int)lp.size(), CV_32FC1);
    for (int i = 0, _end = (int)lp.size(); i < _end; i++ ) {
        g2o::VertexSBAPointXYZ* v = 
                dynamic_cast< g2o::VertexSBAPointXYZ* > 
                ( optimizer.vertex(i+2) );
        Eigen::Vector3d pos = v->estimate();
        pts_4d.at<float>(0, i) = pos[0];
        pts_4d.at<float>(1, i) = pos[1];
        pts_4d.at<float>(2, i) = pos[2];
        pts_4d.at<float>(3, i) = 1;
    }

    // check points 
    bool checkPassed = false;
    TIME_BEGIN()
    checkPassed = CheckPoints(R,t, pts_4d);
    TIME_END("check points")

    if( !checkPassed ) {
        return false;
    }

    // insert map points
    for (int i = 0; i < pts_4d.cols; i++) {
        float w = pts_4d.at<float>(3, i);
        if (w!=0 && inliers[i]==1) {
            cv::Point3f* mpt = new cv::Point3f(
                    pts_4d.at<float>(0, i)/w,
                    pts_4d.at<float>(1, i)/w,
                    pts_4d.at<float>(2, i)/w
                    );
            cout << *mpt << endl;
            map->mapPoints.insert(mpt);
            lf.map_2d_3d.insert(
                    Map_2d_3d_key_val(pt_idx[i], mpt)
                    );
        }
    }

    // insert key frame
    ImageFrame* nkf = new ImageFrame(firstFrame); 
    map->keyFrames.push_back(nkf);
    resFirstFrame = nkf;

    cout << "R: " << endl
        << R << endl;
    cout << "t: " << endl
        << t << endl;

    rf.R = R.clone();
    rf.t = t.clone();

    return true;   
}

bool Initializer::CheckPoints(cv::Mat &R, cv::Mat &t, cv::Mat &pts)
{
    cout << endl << "CheckPoints:" << endl;

    cout << "R:" << endl;
    cout << R << endl;

    cout << "t:" << endl;
    cout << t << endl;

    int inliers = 0;
    // check parallax
    cv::Mat O1O2 = -R.t()*t; 
    for (int i = 0; i < pts.cols; i++) {
        cv::Mat pts_3d(3, 1, CV_64FC1);
        float w = pts.at<float>(3, i);
        pts_3d.at<double>(0) = pts.at<float>(0, i) / w;
        pts_3d.at<double>(1) = pts.at<float>(1, i) / w;
        pts_3d.at<double>(2) = pts.at<float>(2, i) / w;

        if (pts_3d.at<double>(2)<0) {
            pts.at<float>(3, i) = 0;   // mark as outlier
            continue;
        }

        cv::Mat O2P = pts_3d - O1O2;

        //cout << "O1P:" << Converter::getImageType(pts_3d.type()) << endl;
        //cout << pts_3d << endl;

        //cout << "O2P:" << Converter::getImageType(O2P.type()) << endl;
        //cout << O2P << endl;

        double cosAngle = pts_3d.dot(O2P) / (cv::norm(pts_3d) * cv::norm(O2P));

        if (cosAngle > 0.999) {
            //printf("Point disparty too smalll , cosAngle: %f\n", cosAngle);
        } else {
            //printf("Point disparty , cosAngle: %f\n", cosAngle);
            inliers++;
        }
    }

    double ratio = (double)inliers/(double)pts.cols;
    if (ratio < 0.5) {
        printf("Triangulation inliers too small !\n");
        return false;
    }

    printf("Triangulation inliers num: %d ratio: %f\n", inliers, ratio);

    return true;

}
