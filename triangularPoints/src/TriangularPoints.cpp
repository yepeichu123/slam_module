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

    mdfx_ = K.at<float>(0, 0);
    mdfy_ = K.at<float>(1, 1);
    mdcx_ = K.at<float>(0, 2);
    mdcy_ = K.at<float>(1, 2);
}

TriangularPoints::~TriangularPoints() {

}

void TriangularPoints::runTriangularPoints(const TYPE &type,
    const cv::Mat &Pose1, const cv::Mat &Pose2,
    std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2, 
    std::vector<cv::DMatch> &matches, std::vector<float> &depth) {
    if (mbWrong_) {
        cout << "Wrong status. exit!" << endl;
        return;
    }

    switch(type) {
        case TYPE::ORBSLAM: {
            for (int i = 0; i < matches.size(); ++i) {
                cv::KeyPoint p1 = kpt1[matches[i].queryIdx];
                cv::KeyPoint p2 = kpt2[matches[i].trainIdx];
                cv::Mat p3d;
                ORBSLAMTriangular(p1, p2, Pose1, Pose2, p3d);
                depth.push_back(p3d.at<float>(2));
            }
            break;
        }
        case TYPE::VINS_MONO: {
            for (int i = 0; i < matches.size(); ++i) {
                
            }
            break;
        }
        case TYPE::CV: {
            break;
        }
        default: {

        }
    }
}

// the method ORBSLAM used.
void TriangularPoints::ORBSLAMTriangular(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2, 
    const cv::Mat &P1, const cv::Mat &P2, const cv::Mat &X3D) {
    
    // construct A matrix
    cv::Mat A(4, 4, CV_32FC1);

    // x1^ * P1 * X = 0
    // x2^ * P2 * X = 0
    // ==> [x1^P1; x2^P2]*X = 0
    A.row(0) = kpt1.pt.x*P1.row(2) - P1.row(0);
    A.row(1) = kpt1.pt.y*P1.row(2) - P1.row(1);
    A.row(2) = kpt2.pt.x*P2.row(2) - P1.row(0);
    A.row(3) = kpt2.pt.y*P2.row(2) - P1.row(1);

    // the last column of v is the minimum cost of Ax
    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A|cv::SVD::FULL_UV);
    X3D = vt.row(3).t();

    // normalized points 
    X3D = X3D.rowRange(0, 3) / X3D.at<float>(3);
}

// the method VINS-Mono used.
void TriangularPoints::VINSTriangular(Eigen::Matrix<float, 3, 4> &Pose1, Eigen::Matrix<float, 3, 4> &Pose2,
    Eigen::Vector2f &p2d_1, Eigen::Vector2f &p2d_2, Eigen::Vector3f &p3d) {
    
    // construct H matrix
    Eigen::Matrix4f design_matrix = Eigen::Matrix4f::Zero();

    // x1^ * P1 * X = 0
    // x2^ * P2 * X = 0
    // ==> [x1^P1; x2^P2]*X = 0
    design_matrix.row(0) = p2d_1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(1) = p2d_1[1] * Pose1.row(2) - Pose1.row(1);
    design_matrix.row(2) = p2d_2[0] * Pose2.row(2) - Pose2.row(0);
    design_matrix.row(3) = p2d_2[1] * Pose2.row(2) - Pose2.row(1);

    // compute the x which satisfied min(Ax)
    Eigen::Vector4f triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullU).matrixV().rightCols<1>();

    // normalized point
    p3d(0) = triangulated_point(0) / triangulated_point(3);
    p3d(1) = triangulated_point(1) / triangulated_point(3);
    p3d(2) = triangulated_point(2) / triangulated_point(3);
}

// the method OpenCV used 
void TriangularPoints::CVTriangular(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2, 
    const cv::Mat &P1, const cv::Mat &P2, const cv::Mat &X3D) {
    
    // triangulate point
    cv::triangulatePoints(P1, P2, kpt1.pt, kpt2.pt, X3D);

    // normalized point 
    X3D = X3D.rowRange(0, 3) / X3D.at<float>(3);
}