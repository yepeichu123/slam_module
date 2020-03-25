#include "DepthFilter.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>
#include <iostream>

const double PI = 3.14159265358979323846264338328;

DepthFilter::DepthFilter(const cv::Mat &K) {
    mK_ = K.clone();
}

DepthFilter::~DepthFilter() {

}

// run depth filter 
void DepthFilter::RunSingleDepthFilter(Points &p_1, Points &p_2) {
    float u_1 = p_1.GetDepth();
    float u_2 = p_2.GetDepth();

    float s_1 = p_1.GetUncertainty();
    float s_2 = p_2.GetUncertainty();

    float new_depth = (s_1*s_1*u_2 + s_2*s_2*u_1) / (s_1*s_1 + s_2*s_2);
    float new_depth_uncertain = ((s_1*s_1) * (s_2*s_2)) / (s_1*s_1 + s_2*s_2);

    p_1.UpdatePointDepth(new_depth, new_depth_uncertain);
}

// relative pose : [R | t] means from frame 2 to frame 1
void DepthFilter::ComputeTriangulatePoint(const cv::Mat &R, const cv::Mat &t, 
    Points &p_1, Points &p_2) {
    cv::KeyPoint x1 = p_1.GetKeyPoint();
    cv::KeyPoint x2 = p_2.GetKeyPoint();

    // construct transformate matrix 
    std::cout << R << "\n" << t << std::endl;
    cv::Mat T_1 = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat T_2 = cv::Mat::eye(4, 4, CV_32F);
    R.rowRange(0,3).colRange(0,3).copyTo(T_2(cv::Rect(0,0,3,3)));   
    t.rowRange(0,3).copyTo(T_2(cv::Rect(3,0,1,3)));
    std::cout << T_1 << "\n" << T_2 << std::endl;

    // compute project matrix 
    cv::Mat K;
    K = (cv::Mat_<float>(3, 4) << mK_.at<float>(0,0), mK_.at<float>(0,1), mK_.at<float>(0,2), 0,
                                  mK_.at<float>(1,0), mK_.at<float>(1,1), mK_.at<float>(1,2), 0,
                                  mK_.at<float>(2,0), mK_.at<float>(2,1), mK_.at<float>(2,2), 0);
    cv::Mat P_1 = K * T_1;
    cv::Mat P_2 = K * T_2;

    // compute H matrix 
    cv::Mat H(4,4, CV_32F);
    H.row(0) = x1.pt.x * P_1.row(2) - P_1.row(0);
    H.row(1) = x1.pt.y * P_1.row(2) - P_1.row(1);
    H.row(2) = x2.pt.x * P_2.row(2) - P_2.row(0);
    H.row(3) = x2.pt.y * P_2.row(2) - P_2.row(1);
    cv::Mat w, u, vt;
    cv::SVD::compute(H, w, u, vt, cv::SVD::MODIFY_A|cv::SVD::FULL_UV);
    cv::Mat X3d = vt.row(3).t();

    // normalized point
    X3d = X3d.rowRange(0,3) / X3d.at<float>(3);
    float d = X3d.at<float>(2);
    if (d > 0) {
        p_1.SetPointDepth(d);
    }
    else {
        p_1.SetPointDepth(0);
    }
}

// compute uncertainty
void DepthFilter::ComputeUncertainty(const cv::Mat &R, const cv::Mat &t, 
    Points &p_1, Points &p_2) {
    cv::KeyPoint x1 = p_1.GetKeyPoint();
    cv::KeyPoint x2 = p_2.GetKeyPoint();
    float d = p_1.GetDepth();
    cv::Mat P = (cv::Mat_<float>(3,1) << (x1.pt.x - mK_.at<float>(0, 2)) * d / mK_.at<float>(0, 0),
                                         (x1.pt.y - mK_.at<float>(1, 2)) * d / mK_.at<float>(1, 1),
                                          d);
    cv::Mat a = P - t;
    float alpha = std::acos(P.dot(t) / (cv::norm(P) * cv::norm(t)));
    float beta = std::acos(a.dot(-t) / (cv::norm(a) * cv::norm(-t)));
    float delta_beta = std::atan(1.0 / mK_.at<float>(0,0));
    float new_beta = beta - delta_beta;
    float gamma = PI - alpha - new_beta;
    
    float new_depth = cv::norm(t) * std::sin(new_beta) / std::sin(gamma);
    float depth_uncertain = p_1.GetDepth() - new_depth;
    p_1.SetDepthUncertainty(depth_uncertain);
    p_2.SetPointDepth(cv::norm(a));
    p_2.SetDepthUncertainty(depth_uncertain);
}