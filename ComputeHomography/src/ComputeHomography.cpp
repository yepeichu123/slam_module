#include "ComputeHomography.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;

ComputeHomography::ComputeHomography(const cv::Mat &K) {
    if (K.empty() || K.rows != 3 || K.cols != 3) {
        cout << "invalid camera intrinsics, please check it again." << endl;
        return;
    }
    mK_ = K.clone();
}

ComputeHomography::~ComputeHomography() {
    
}

void ComputeHomography::RunComputeHomography(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
    std::vector<cv::DMatch> &matches, cv::Mat &H) {
    std::vector<cv::Point2f> ref_cam, cur_cam;
    for (int i = 0; i < matches.size(); ++i) {
        cv::Point2f p1 = ref_kpts[matches[i].queryIdx].pt;
        cv::Point2f p2 = cur_kpts[matches[i].trainIdx].pt;
        /*cv::Point2f pixel_1(
            (p1.x - mK_.at<float>(0, 2)) / mK_.at<float>(0, 0),
            (p1.y - mK_.at<float>(1, 2)) / mK_.at<float>(1, 1)
        );
        cv::Point2f pixel_2(
            (p2.x - mK_.at<float>(0, 2)) / mK_.at<float>(0, 0),
            (p2.y - mK_.at<float>(1, 2)) / mK_.at<float>(1, 1)
        );*/
        ref_cam.push_back(p1);
        cur_cam.push_back(p2);
    }

    cv::Mat inliers;
    cv::Mat new_H = cv::findHomography(ref_cam, cur_cam, cv::RANSAC, 3.0, inliers, 2000, 0.995);
    new_H.convertTo(H, CV_32F);
    cout << "We compute the homography matrix :\n" << H << endl;
    cout << "Before RANSAC : " << endl;
    CheckPoseError(ref_kpts, cur_kpts, matches, H);

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < matches.size(); ++i) {
        if (inliers.at<int>(i) != 0) {
            good_matches.push_back(matches[i]);
        }
    }
    matches.clear();
    matches.insert(matches.end(), good_matches.begin(), good_matches.end());
    cout << "After RANSAC : " << endl;
    CheckPoseError(ref_kpts, cur_kpts, matches, H);
    cout << "Finally, we get " << matches.size() << " good matching pairs!" << endl;  
}

float ComputeHomography::CheckPoseError(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
    const std::vector<cv::DMatch> &matches, const cv::Mat &H) {
    
    float error = 0;
    for (int i = 0; i < matches.size(); ++i) {
        cv::Point2f p1 = ref_kpts[matches[i].queryIdx].pt;
        cv::Point2f p2 = cur_kpts[matches[i].trainIdx].pt;

        cv::Mat p1_cam = (cv::Mat_<float>(3, 1) <<  (p1.x - mK_.at<float>(0, 2)) / mK_.at<float>(0, 0),
                                                    (p1.y - mK_.at<float>(1, 2)) / mK_.at<float>(1, 1),
                                                    1);
        cv::Mat p2_cam = (cv::Mat_<float>(3, 1) <<  (p2.x - mK_.at<float>(0, 2)) / mK_.at<float>(0, 0),
                                                    (p2.y - mK_.at<float>(1, 2)) / mK_.at<float>(1, 1),
                                                    1);

        cv::Mat e = p2_cam - H * p1_cam;
        error += cv::norm(e);
    }
    cout << "Average error is " << error << endl;

    return error;
}

void ComputeHomography::ImagesStitch(const cv::Mat &ref_img, const cv::Mat &cur_img, const cv::Mat &H, cv::Mat &out_img) {
    
    cv::Mat canvas;
    cv::warpPerspective(ref_img, canvas, H, cv::Size(cur_img.cols*2, cur_img.rows));
    cur_img.copyTo(canvas(cv::Range::all(), cv::Range(0, cur_img.cols)));

    canvas.copyTo(out_img);
}