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

void TriangularPoints::runTriangularPoints(const TYPE &type, const bool &proj_pose,
    const cv::Mat &Pose1, const cv::Mat &Pose2,
    std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2, 
    std::vector<cv::DMatch> &matches, std::vector<cv::Point3f> &points3d) {
    if (mbWrong_) {
        cout << "Wrong status. exit!" << endl;
        return;
    }

    if (!proj_pose) {
        cv::Mat K;
        K = (cv::Mat_<float>(3, 4) << mffx_, 0, mfcx_, 0,
                                      0, mffy_, mfcy_, 0,
                                      0, 0, 1, 0);
        mmP1_ = K * Pose1;
        mmP2_ = K * Pose2;
    }
    else {
        mmP1_ = Pose1;
        mmP2_ = Pose2;
    }

    switch(type) {
        case TYPE::ORBSLAM: {
            cout << "ORBSLAM-Triangualte points!" << endl;
            for (int i = 0; i < matches.size(); ++i) {
                cv::KeyPoint p1 = kpt1[matches[i].queryIdx];
                cv::KeyPoint p2 = kpt2[matches[i].trainIdx];
                cv::Mat p3d;
                ORBSLAMTriangular(p1, p2, mmP1_, mmP2_, p3d);
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
            Eigen::Matrix<float, 3, 4> P1, P2;
            P1 << mmP1_.at<float>(0,0), mmP1_.at<float>(0,1), mmP1_.at<float>(0,2), mmP1_.at<float>(0,3),
                    mmP1_.at<float>(1,0), mmP1_.at<float>(1,1), mmP1_.at<float>(1,2), mmP1_.at<float>(1,3),
                    mmP1_.at<float>(2,0), mmP1_.at<float>(2,1), mmP1_.at<float>(2,2), mmP1_.at<float>(2,3);
                    
            P2 << mmP2_.at<float>(0,0), mmP2_.at<float>(0,1), mmP2_.at<float>(0,2), mmP2_.at<float>(0,3),
                    mmP2_.at<float>(1,0), mmP2_.at<float>(1,1), mmP2_.at<float>(1,2), mmP2_.at<float>(1,3),
                    mmP2_.at<float>(2,0), mmP2_.at<float>(2,1), mmP2_.at<float>(2,2), mmP2_.at<float>(2,3);

            cout << "P1 = " << P1 << "\n P2 = " << P2 << endl;
            for (int i = 0; i < matches.size(); ++i) {
                cv::KeyPoint p1 = kpt1[matches[i].queryIdx];
                cv::KeyPoint p2 = kpt2[matches[i].trainIdx];
                Eigen::Vector2f p2d_1(p1.pt.x, p1.pt.y);
                Eigen::Vector2f p2d_2(p2.pt.x, p2.pt.y);
                Eigen::Vector3f p3d;
                VINSTriangular(p2d_1, p2d_2, P1, P2, p3d);
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
            CVTriangular(matches, kpt1, kpt2, mmP1_, mmP2_, points3d);
            break;
        }
        default: {
            cout << "Default-Triangulate points!" << endl;
            for (int i = 0; i < matches.size(); ++i) {
                cv::KeyPoint p1 = kpt1[matches[i].queryIdx];
                cv::KeyPoint p2 = kpt2[matches[i].trainIdx];
                cv::Mat p3d;
                DefaultTriangular(p1, p2, mmP1_, mmP2_, p3d);
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
void TriangularPoints::ORBSLAMTriangular(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2, 
    const cv::Mat &P1, const cv::Mat &P2, cv::Mat &X3D) {
    
    // construct A matrix
    cv::Mat A(4, 4, CV_32F);

    // x1^ * P1 * X = 0
    // x2^ * P2 * X = 0
    // ==> [x1^P1; x2^P2]*X = 0
    A.row(0) = kpt1.pt.x*P1.row(2) - P1.row(0);
    A.row(1) = kpt1.pt.y*P1.row(2) - P1.row(1);
    A.row(2) = kpt2.pt.x*P2.row(2) - P2.row(0);
    A.row(3) = kpt2.pt.y*P2.row(2) - P2.row(1);

    // the last column of v is the minimum cost of Ax
    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A|cv::SVD::FULL_UV);
    X3D = vt.row(3).t();

    // normalized points 
    X3D = X3D.rowRange(0, 3) / X3D.at<float>(3);
}

// the method VINS-Mono used.
void TriangularPoints::VINSTriangular(Eigen::Vector2f &p2d_1, Eigen::Vector2f &p2d_2,
    Eigen::Matrix<float, 3, 4> &P1, Eigen::Matrix<float, 3, 4> &P2, Eigen::Vector3f &p3d) {
    
    // construct H matrix
    Eigen::Matrix4f design_matrix = Eigen::Matrix4f::Zero();

    // x1^ * P1 * X = 0
    // x2^ * P2 * X = 0
    // ==> [x1^P1; x2^P2]*X = 0
    design_matrix.row(0) = p2d_1[0] * P1.row(2) - P1.row(0);
    design_matrix.row(1) = p2d_1[1] * P1.row(2) - P1.row(1);
    design_matrix.row(2) = p2d_2[0] * P2.row(2) - P2.row(0);
    design_matrix.row(3) = p2d_2[1] * P2.row(2) - P2.row(1);

    // compute the x which satisfied min(Ax)
    Eigen::Vector4f triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    // normalized point
    p3d(0) = triangulated_point(0) / triangulated_point(3);
    p3d(1) = triangulated_point(1) / triangulated_point(3);
    p3d(2) = triangulated_point(2) / triangulated_point(3);
}

// the method OpenCV used 
void TriangularPoints::CVTriangular(std::vector<cv::DMatch> &matches,
    const std::vector<cv::KeyPoint> &kpt1, const std::vector<cv::KeyPoint> &kpt2, 
    const cv::Mat &P1, const cv::Mat &P2, std::vector<cv::Point3f> &points3d) {
    
    // transform from pixel coordinate to camera coordinate
    std::vector<cv::Point2f> p2d_1, p2d_2;
    for (int i = 0; i < matches.size(); ++i) {
        p2d_1.push_back(kpt1[matches[i].queryIdx].pt);
        p2d_2.push_back(kpt2[matches[i].trainIdx].pt);
    }

    cv::Mat pst_4d;
    // triangulate point
    cv::triangulatePoints(P1, P2, p2d_1, p2d_2, pst_4d);

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

void TriangularPoints::DefaultTriangular(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2,
    const cv::Mat &P1, const cv::Mat &P2, cv::Mat &X3D) {
    
    cv::Point2f p1 = kpt1.pt;
    cv::Point2f p2 = kpt2.pt;

    // x1^ * P1 * X = 0
    // x2^ * P2 * X = 0
    // ==> [x1^P1; x2^P2]*X = 0
    cv::Mat H(4, 4, CV_32F);
    H.row(0) = p1.x * P1.row(2) - P1.row(0);
    H.row(1) = -p1.y * P1.row(2) + P1.row(1);
    // H.row(1) = p1.x * P1.row(1) - p1.y * P1.row(0);
    H.row(2) = p2.x * P2.row(2) - P2.row(0);
    H.row(3) = -p2.y * P2.row(2) + P2.row(1);
    // H.row(3) = p2.x * P2.row(1) - p2.y * P2.row(0);

    // SVD
    cv::Mat u, w, vt;
    cv::SVD::compute(H, w, u, vt, cv::SVD::MODIFY_A|cv::SVD::FULL_UV);
    X3D = vt.row(3).t();

    // normalized points 
    X3D = X3D.rowRange(0, 3) / X3D.at<float>(3);
}