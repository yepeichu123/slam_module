#include "EpipolarConstrain.h"
#include <vector>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>

#define PI 3.1415926
using namespace std;

EpipolarConstrain::EpipolarConstrain(const cv::Mat &K) {
    mK_ = K.clone();
}

EpipolarConstrain::~EpipolarConstrain() {

}

bool EpipolarConstrain::ComputeRelativePose(std::vector<cv::KeyPoint> &ref_kpts, std::vector<cv::KeyPoint> &cur_kpts, 
            std::vector<cv::DMatch> &matches, cv::Mat &R, cv::Mat &t) {
    
    vector<cv::Point2f> ref_pixel, cur_pixel;
    for (int i = 0; i < matches.size(); ++i) {
        ref_pixel.push_back(ref_kpts[matches[i].queryIdx].pt);
        cur_pixel.push_back(cur_kpts[matches[i].trainIdx].pt);
    }

    cv::Mat inliers;
    cv::Mat E = cv::findEssentialMat(ref_pixel, cur_pixel, mK_, 8, 0.999, 1.0, inliers);
    cv::Mat rvec, tvec;
    cv::recoverPose(E, ref_pixel, cur_pixel, mK_, rvec, tvec, inliers);
    rvec.convertTo(R, CV_32F);
    tvec.convertTo(t, CV_32F);
    
    vector<cv::KeyPoint> ref_p, cur_p;
    vector<cv::DMatch> good_matches;
    int count = 0;
    for (int i = 0; i < inliers.rows; ++i) {
        
        if (inliers.at<int>(i) != 0) {
            ref_p.push_back(ref_kpts[matches[i].queryIdx]);
            cur_p.push_back(cur_kpts[matches[i].trainIdx]);
            cv::DMatch temp_match;
            temp_match.queryIdx = count;
            temp_match.trainIdx = count;
            good_matches.push_back(temp_match);
            ++count;
        }
    }
    cout << "After RANSAC, we have " << ref_p.size() << " matching pairs!" << endl;

    ref_kpts.clear();
    cur_kpts.clear();
    matches.clear();
    ref_kpts.insert(ref_kpts.end(), ref_p.begin(), ref_p.end());
    cur_kpts.insert(cur_kpts.end(), cur_p.begin(), cur_p.end());
    matches.insert(matches.end(), good_matches.begin(), good_matches.end());
}

int EpipolarConstrain::MatchingByRANSAC() {

}

float EpipolarConstrain::CheckPoseError() {

}

void EpipolarConstrain::EpipolarConstrainFor8Points(std::vector<cv::Point2f> &ref_pixel, std::vector<cv::Point2f> &cur_pixel,
    cv::Mat &E) {
    if ((ref_pixel.size() != 8 || cur_pixel.size() != 8) && ref_pixel.size() != cur_pixel.size()) {
        cout << "Please make sure we have 8 pair points!" << endl;
        return;
    }

    E = cv::Mat::zeros(9, 1, CV_32F);
    E.at<float>(8) = 1;
    cv::Mat A = cv::Mat::zeros(8, 9, CV_32F);
    for (int i = 0; i < ref_pixel.size(); ++i) {

        cv::Point2f ref_cam(
            (ref_pixel[i].x - mK_.at<float>(0, 2)) / mK_.at<float>(0, 0),
            (ref_pixel[i].y - mK_.at<float>(1, 2)) / mK_.at<float>(1, 1)
        );
        cv::Point2f cur_cam(
            (cur_pixel[i].x - mK_.at<float>(0, 2)) / mK_.at<float>(0, 0),
            (cur_pixel[i].y - mK_.at<float>(1, 2)) / mK_.at<float>(1, 1)
        );

        float u1 = ref_cam.x;
        float v1 = ref_cam.y;
        float u2 = cur_cam.x;
        float v2 = cur_cam.y;
        A.row(i) = (cv::Mat_<float>(1, 9) << u1*u2, u1*v2, u1, u2*v1, v1*v2, v1, u2, v2, 1);
    }

    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A|cv::SVD::FULL_UV);
    cout << "vt = " << vt << endl;
    E = vt.row(8).t();
    cout << "E = " << E << endl;

    cv::Mat E_new = (cv::Mat_<float>(3, 3) << E.at<float>(0), E.at<float>(1), E.at<float>(2),
                                              E.at<float>(3), E.at<float>(4), E.at<float>(5),
                                              E.at<float>(6), E.at<float>(7), E.at<float>(8));
    E = E_new.clone();

    cv::Mat R, t;
    FindBestPoseFromEssMat(E, ref_pixel, cur_pixel, R, t);
}

void EpipolarConstrain::FindBestPoseFromEssMat(const cv::Mat &E, std::vector<cv::Point2f> &ref_pixel, std::vector<cv::Point2f> &cur_pixel,
    cv::Mat &R, cv::Mat &t) {
    if (E.empty()) {
        cout << "E is empty! return." << endl;
        return;
    }

    cv::Mat u, w, vt;
    cv::SVD::compute(E, w, u, vt, cv::SVD::FULL_UV);
    cout << "u = " << u << endl;
    cout << "w = " << w << endl;
    cout << "vt = " << vt << endl;
    cv::Mat S = (cv::Mat_<float>(3,3) << w.at<float>(0), 0, 0,
                                         0, w.at<float>(1), 0,
                                         0, 0, w.at<float>(2));


    cv::Mat rot1, rot2;
    cv::Mat rvec1 = (cv::Mat_<float>(3,1) << 0, 0, -PI/2);
    cv::Mat rvec2 = (cv::Mat_<float>(3,1) << 0, 0, PI/2);
    cv::Rodrigues(rvec1, rot1);
    cv::Rodrigues(rvec2, rot2);

    cv::Mat tx1, tx2, tx3, tx4;
    cv::Mat r1, r2, r3, r4;
    // cond 1
    tx1 = u * rot1 * S * u.t();
    r1 = u * rot1.t() * vt;
    // cond2
    tx2 = u * rot2 * S * u.t();
    r2 = u * rot2.t() * vt;
    // cond3
    tx3 = -tx1;
    r3 = r1;
    // cond4
    tx4 = -tx2;
    r4 = r2;

    cout << "tx1 = " << tx1 << endl;
    cout << "tx2 = " << tx2 << endl;
}
