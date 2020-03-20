#include "TriangularPoints.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

using namespace std;

TriangularPoints::TriangularPoints(const cv::Mat &K) {
    if (K.empty()) {
        cout << "The camera intrinsics are empty, exit." << endl;
        mbWrong_ = true;
        return;
    }
    mbWrong_ = false;

    mffx_ = K.at<float>(0, 0);
    mffy_ = K.at<float>(1, 1);
    mfcx_ = K.at<float>(0, 2);
    mfcy_ = K.at<float>(1, 2);
}

TriangularPoints::~TriangularPoints() {
}

void TriangularPoints::runTriangularPoints(const TYPE &type,
    const cv::Mat &Pose1, const cv::Mat &Pose2,
    std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2, 
    std::vector<cv::DMatch> &matches, std::vector<cv::Point3f> &points3d) {

    if (mbWrong_) {
        cout << "Wrong status. exit!" << endl;
        return;
    }

    ConstructProjectMatrix_(Pose1, Pose2);

    switch(type) {
        case TYPE::ORBSLAM: {
            cout << "ORBSLAM-Triangualte points!" << endl;
            for (int i = 0; i < matches.size(); ++i) {
                cv::KeyPoint p1 = kpt1[matches[i].queryIdx];
                cv::KeyPoint p2 = kpt2[matches[i].trainIdx];
                cv::Mat p3d;
                ORBSLAMTriangular_(p1, p2, p3d);
                float d = p3d.at<float>(2);
                if (d < 0) {
                    continue;
                }
                cv::Point3f point3d(p3d.at<float>(0), p3d.at<float>(1), p3d.at<float>(2));
                points3d.push_back(point3d);
            }
            break;
        }
        case TYPE::VINS_MONO: {
            cout << "VINS_MONO-Triangualte points!" << endl;
            ConvertMatToEigen_();

            for (int i = 0; i < matches.size(); ++i) {
                cv::KeyPoint p1 = kpt1[matches[i].queryIdx];
                cv::KeyPoint p2 = kpt2[matches[i].trainIdx];
                Eigen::Vector2f p2d_1(p1.pt.x, p1.pt.y);
                Eigen::Vector2f p2d_2(p2.pt.x, p2.pt.y);
                Eigen::Vector3f p3d;
                VINSTriangular_(p2d_1, p2d_2, p3d);
                float d = p3d[2];
                if (d < 0) {
                    continue;
                }
                cv::Point3f p(p3d[0], p3d[1], p3d[2]);
                points3d.push_back(p);
            }
            break;
        }
        case TYPE::CV: {
            cout << "OpenCV-Triangualte points!" << endl;
            CVTriangular_(matches, kpt1, kpt2, points3d);
            break;
        }
        default: {
            cout << "Default-Triangulate points!" << endl;
            for (int i = 0; i < matches.size(); ++i) {
                cv::KeyPoint p1 = kpt1[matches[i].queryIdx];
                cv::KeyPoint p2 = kpt2[matches[i].trainIdx];
                cv::Mat p3d;
                DefaultTriangular_(p1, p2, p3d);
                float d = p3d.at<float>(2);
                if (d < 0) {
                    continue;
                }
                cv::Point3f point3d(p3d.at<float>(0), p3d.at<float>(1), p3d.at<float>(2));
                points3d.push_back(point3d);
            }
            break;
        }
    }
}

// the method ORBSLAM used.
void TriangularPoints::ORBSLAMTriangular_(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2, cv::Mat &X3D) {
    
    // construct A matrix
    cv::Mat A(4, 4, CV_32F);

    // x1^ * P1 * X = 0
    // x2^ * P2 * X = 0
    // ==> [x1^P1; x2^P2]*X = 0
    A.row(0) = kpt1.pt.x*mmP1_.row(2) - mmP1_.row(0);
    A.row(1) = kpt1.pt.y*mmP1_.row(2) - mmP1_.row(1);
    A.row(2) = kpt2.pt.x*mmP2_.row(2) - mmP2_.row(0);
    A.row(3) = kpt2.pt.y*mmP2_.row(2) - mmP2_.row(1);

    // the last column of v is the minimum cost of Ax
    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A|cv::SVD::FULL_UV);
    X3D = vt.row(3).t();

    // normalized points 
    X3D = X3D.rowRange(0, 3) / X3D.at<float>(3);
}

// the method VINS-Mono used.
void TriangularPoints::VINSTriangular_(Eigen::Vector2f &p2d_1, Eigen::Vector2f &p2d_2, Eigen::Vector3f &p3d) {
    
    // construct H matrix
    Eigen::Matrix4f design_matrix = Eigen::Matrix4f::Zero();

    // x1^ * P1 * X = 0
    // x2^ * P2 * X = 0
    // ==> [x1^P1; x2^P2]*X = 0
    design_matrix.row(0) = p2d_1[0] * meP1_.row(2) - meP1_.row(0);
    design_matrix.row(1) = p2d_1[1] * meP1_.row(2) - meP1_.row(1);
    design_matrix.row(2) = p2d_2[0] * meP2_.row(2) - meP2_.row(0);
    design_matrix.row(3) = p2d_2[1] * meP2_.row(2) - meP2_.row(1);

    // compute the x which satisfied min(Ax)
    Eigen::Vector4f triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    // normalized point
    p3d(0) = triangulated_point(0) / triangulated_point(3);
    p3d(1) = triangulated_point(1) / triangulated_point(3);
    p3d(2) = triangulated_point(2) / triangulated_point(3);
}

// the method OpenCV used 
void TriangularPoints::CVTriangular_(std::vector<cv::DMatch> &matches,
    const std::vector<cv::KeyPoint> &kpt1, const std::vector<cv::KeyPoint> &kpt2,std::vector<cv::Point3f> &points3d) {
    
    // transform from pixel coordinate to camera coordinate
    std::vector<cv::Point2f> p2d_1, p2d_2;
    for (int i = 0; i < matches.size(); ++i) {
        p2d_1.push_back(kpt1[matches[i].queryIdx].pt);
        p2d_2.push_back(kpt2[matches[i].trainIdx].pt);
    }

    cv::Mat pst_4d;
    // triangulate point
    cv::triangulatePoints(mmP1_, mmP2_, p2d_1, p2d_2, pst_4d);

    // normalized point 
    for (int i = 0; i < pst_4d.cols; ++i) {
        cv::Mat x3d = pst_4d.col(i);
        x3d = x3d.rowRange(0,3) / x3d.at<float>(3);
        float d = x3d.at<float>(2);
        if (d < 0) {
            continue;
        }
        cv::Point3f p3d(x3d.at<float>(0), x3d.at<float>(1), x3d.at<float>(2));
        points3d.push_back(p3d);
    }
}

void TriangularPoints::DefaultTriangular_(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2, cv::Mat &X3D) {

    // convert pixel points to normalized plane points 
    cv::Point2f pixel_1 = kpt1.pt;
    cv::Point2f pixel_2 = kpt2.pt;
    cv::Mat p1, p2;
    p1 = (cv::Mat_<float>(3,1) << (pixel_1.x - mfcx_) / mffx_,
                                (pixel_1.y - mfcy_) / mffy_,
                                1);
    p2 = (cv::Mat_<float>(3,1) << (pixel_2.x - mfcx_) / mffx_,
                                (pixel_2.y - mfcy_) / mffy_,
                                1);

    // extract relative pose between two images 
    cv::Mat T, R, t;
    T = mmPose1_ * mmPose2_.inv();
    R = T(cv::Range(0,3), cv::Range(0,3));
    t = T(cv::Range(0,3), cv::Range(3,4));

    // here we have Ax = B
    // construct A matrix 
    float A[4];
    A[0] = p1.dot(p1);
    A[2] = p1.dot(R * p2);
    A[1] = -A[2];
    A[3] = -(R * p2).dot(R * p2);
    // construct B matrix 
    float B[2];
    B[0] = t.dot(p1);
    B[1] = t.dot(R * p2);

    // Cramer's rule 
    float A_det = A[0]*A[3] - A[1]*A[2];
    float z1 = (A[3] * B[0] - A[1] * B[1]) / A_det;
    float z2 = (-A[2] * B[0] + A[0] * B[1]) / A_det;

    // compute depth 
    cv::Mat X1 = z1 * p1;
    cv::Mat X2 = z2 * R * p2 + t;
    cv::Mat d_esti = (X1 + X2) / 2.0;
    float depth = cv::norm(d_esti);
    // cout << "depth = " << depth << endl;

    // return the reference point depth
    X3D = p1 * depth;
}

void TriangularPoints::ConstructProjectMatrix_(const cv::Mat &Pose1, const cv::Mat &Pose2) {
         
        // camera pose  
        mmPose1_ = Pose1.clone();
        mmPose2_ = Pose2.clone();
        
        // camera instrinsic
        cv::Mat K;
        K = (cv::Mat_<float>(3, 4) << mffx_, 0, mfcx_, 0,
                                      0, mffy_, mfcy_, 0,
                                      0, 0, 1, 0);
        // camera project matrix
        mmP1_ = K * Pose1;
        mmP2_ = K * Pose2;
} 

void TriangularPoints::ConvertMatToEigen_() { 

    meP1_ << mmP1_.at<float>(0,0), mmP1_.at<float>(0,1), mmP1_.at<float>(0,2), mmP1_.at<float>(0,3),
            mmP1_.at<float>(1,0), mmP1_.at<float>(1,1), mmP1_.at<float>(1,2), mmP1_.at<float>(1,3),
            mmP1_.at<float>(2,0), mmP1_.at<float>(2,1), mmP1_.at<float>(2,2), mmP1_.at<float>(2,3);
            
    meP2_ << mmP2_.at<float>(0,0), mmP2_.at<float>(0,1), mmP2_.at<float>(0,2), mmP2_.at<float>(0,3),
            mmP2_.at<float>(1,0), mmP2_.at<float>(1,1), mmP2_.at<float>(1,2), mmP2_.at<float>(1,3),
            mmP2_.at<float>(2,0), mmP2_.at<float>(2,1), mmP2_.at<float>(2,2), mmP2_.at<float>(2,3);

}

void TriangularPoints::ComputeAntisymmetricMatrix_(const cv::Mat &p, cv::Mat &matrix) {
    
    float x = p.at<float>(0);
    float y = p.at<float>(1);
    float z = p.at<float>(2);

    matrix = (cv::Mat_<float>(3,3) << 0, -z, y,
                                      z, 0, -x,
                                      -y, x, 0);
}
