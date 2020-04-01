#include "ComputeHomography.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>
#include <fstream>
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
    
    ComputeHomographyMat_(ref_kpts, cur_kpts, matches, H);
    cout << "We compute the homography matrix :\n" << H << endl;

    cout << "After RANSAC : " << endl;
    CheckPoseError(ref_kpts, cur_kpts, matches, H);

    vector<cv::KeyPoint> ref_p, cur_p;
    if (matches.size() > 4) {
        for (int i = 0; i < matches.size(); ++i) {
            ref_p.push_back(ref_kpts[matches[i].queryIdx]);
            cur_p.push_back(cur_kpts[matches[i].trainIdx]);
        }
        cout << "In self-designed method, we compute homography matrix!" << endl;
        cv::Mat HH;
        // CalcHomoMatWithK(ref_p, cur_p, HH);
        // CheckPoseErrorWithK(ref_kpts, cur_kpts, matches, HH);
        CalcHomoMatWithoutK(ref_p, cur_p, HH);
        CheckPoseError(ref_kpts, cur_kpts, matches, HH);
    }
}

float ComputeHomography::CheckPoseError(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
    const std::vector<cv::DMatch> &matches, const cv::Mat &H) {
    
    float error = 0;
    for (int i = 0; i < matches.size(); ++i) {
        cv::Point2f p1 = ref_kpts[matches[i].queryIdx].pt;
        cv::Point2f p2 = cur_kpts[matches[i].trainIdx].pt;
        cv::Mat p1_cam = (cv::Mat_<float>(3, 1) << p1.x, p1.y, 1);
        cv::Mat p2_cam = (cv::Mat_<float>(3, 1) << p2.x, p2.y, 1);
        cv::Mat Hp1 = H * p1_cam;
        cv::Mat Hp1_norm = Hp1 / Hp1.at<float>(2);
        cv::Mat e = p2_cam - Hp1_norm;
        error += cv::norm(e);
    }
    cout << "Average error is " << error / matches.size() << endl;

    return error;
}

float ComputeHomography::CheckPoseErrorWithK(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
    const std::vector<cv::DMatch> &matches, const cv::Mat &H) {
    float error = 0;
    for (int i = 0; i < matches.size(); ++i) {
        cv::Point3f p1, p2;
        Pixel2Cam_(ref_kpts[matches[i].queryIdx], p1);
        Pixel2Cam_(cur_kpts[matches[i].trainIdx], p2);
        cv::Mat p1_cam = (cv::Mat_<float>(3, 1) << p1.x, p1.y, 1);
        cv::Mat p2_cam = (cv::Mat_<float>(3, 1) << p2.x, p2.y, 1);
        cv::Mat Hp1 = H * p1_cam;
        cv::Mat Hp1_norm = Hp1 / Hp1.at<float>(2);
        cv::Mat e = p2_cam - Hp1_norm;
        error += cv::norm(e);
    }
    cout << "Average error is " << error / matches.size() << endl;

    return error;
}

void ComputeHomography::ImagesStitch(const cv::Mat &ref_img, const cv::Mat &cur_img, const cv::Mat &H, cv::Mat &out_img) {
    
    cv::Mat canvas;
    cv::warpPerspective(ref_img, canvas, H, cv::Size(ref_img.cols+cur_img.cols, ref_img.rows));
    cur_img.copyTo(canvas(cv::Range::all(), cv::Range(0, cur_img.cols)));

    // crop image
    CropEmptyImage_(canvas, out_img);
    // canvas.copyTo(out_img);
}

// with K means we should use normalized plane points 
void ComputeHomography::CalcHomoMatWithK(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
    cv::Mat &H) {
    if (ref_kpts.size() < 4 || cur_kpts.size() < 4 || ref_kpts.size() != cur_kpts.size()) {
        cout << "Wrong matching pairs, we need 4 points at least and equal size of two groups." << endl;
        return;
    }

    vector<cv::Point3f> ref_cam, cur_cam;
    Pixel2CamVec_(ref_kpts, ref_cam);
    Pixel2CamVec_(cur_kpts, cur_cam);

    cv::Mat A = cv::Mat::zeros(ref_cam.size() * 2, 8, CV_32F);
    cv::Mat b = cv::Mat::zeros(ref_cam.size() * 2, 1, CV_32F);
    cv::Mat x;
    for (int i = 0; i < ref_cam.size(); ++i) {
        cv::Mat row_1 = (cv::Mat_<float>(1, 8) << ref_cam[i].x, ref_cam[i].y, 1, 0, 0, 0, -ref_cam[i].x*cur_cam[i].x, -ref_cam[i].y*cur_cam[i].x);
        cv::Mat row_2 = (cv::Mat_<float>(1, 8) << 0, 0, 0, ref_cam[i].x, ref_cam[i].y, 1, -ref_cam[i].x*cur_cam[i].y, -ref_cam[i].y*cur_cam[i].y);
        
        row_1.copyTo(A.row(2*i));
        row_2.copyTo(A.row(2*i+1));

        b.at<float>(2*i) = cur_cam[i].x;
        b.at<float>(2*i+1) = cur_cam[i].y;
    }

    cv::solve(A, b, x, cv::DECOMP_SVD);
    if (x.rows == 8) {
        cv::Mat HH = (cv::Mat_<float>(3,3) << x.at<float>(0), x.at<float>(1), x.at<float>(2),
                                            x.at<float>(3), x.at<float>(4), x.at<float>(5),
                                            x.at<float>(6), x.at<float>(7), 1);
        HH.convertTo(H, CV_32F);
        cout << "H = " << H << endl;
    }
}

// without K means we should use pixel plane points (original plane which OpenCV used)
void ComputeHomography::CalcHomoMatWithoutK(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
    cv::Mat &H) {
    if (ref_kpts.size() < 4 || cur_kpts.size() < 4 || ref_kpts.size() != cur_kpts.size()) {
        cout << "Wrong matching pairs, we need 4 points at least and equal size of two groups." << endl;
        return;
    }

    vector<cv::Point2f> ref_cam, cur_cam;
    for (int i = 0; i < ref_kpts.size(); ++i) {
        ref_cam.push_back(ref_kpts[i].pt);
        cur_cam.push_back(cur_kpts[i].pt);
    }

    cv::Mat A = cv::Mat::zeros(ref_cam.size() * 2, 9, CV_32F);
    for (int i = 0; i < ref_cam.size(); ++i) {
        cv::Mat row_1 = (cv::Mat_<float>(1, 9) << ref_cam[i].x, ref_cam[i].y, 1, 0, 0, 0, -ref_cam[i].x*cur_cam[i].x, -ref_cam[i].y*cur_cam[i].x, -cur_cam[i].x);
        cv::Mat row_2 = (cv::Mat_<float>(1, 9) << 0, 0, 0, ref_cam[i].x, ref_cam[i].y, 1, -ref_cam[i].x*cur_cam[i].y, -ref_cam[i].y*cur_cam[i].y, -cur_cam[i].y);
        
        row_1.copyTo(A.row(2*i));
        row_2.copyTo(A.row(2*i+1));
    }

    cv::Mat x;
    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A|cv::SVD::FULL_UV);
    x = vt.row(8).t();

    if (x.rows == 9) {
        float det = cv::norm(x);
        cv::Mat x_norm = x / det;
        x = x_norm / x_norm.at<float>(8);
        cv::Mat HH = (cv::Mat_<float>(3,3) << x.at<float>(0), x.at<float>(1), x.at<float>(2),
                                              x.at<float>(3), x.at<float>(4), x.at<float>(5),
                                              x.at<float>(6), x.at<float>(7), x.at<float>(8));
        HH.convertTo(H, CV_32F);
        cout << "H = " << H << endl;
    }
}

void ComputeHomography::ComputeHomographyMat_(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
    std::vector<cv::DMatch> &matches, cv::Mat &H) {
    std::vector<cv::Point2f> ref_cam, cur_cam;
    for (int i = 0; i < matches.size(); ++i) {
        cv::Point2f p1 = ref_kpts[matches[i].queryIdx].pt;
        cv::Point2f p2 = cur_kpts[matches[i].trainIdx].pt;
        ref_cam.push_back(p1);
        cur_cam.push_back(p2);
    }

    std::vector<char> inliers;
    cv::Mat new_H = cv::findHomography(ref_cam, cur_cam, cv::RANSAC, 3.0, inliers, 2000, 0.99);
    new_H.convertTo(H, CV_32F);

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < inliers.size(); ++i) {
        if (inliers[i] != 0) {
            good_matches.push_back(matches[i]);
        }
    }

    if (good_matches.size() > 10) {
        matches.clear();
        matches.insert(matches.end(), good_matches.begin(), good_matches.end());
    }
}

void ComputeHomography::CropEmptyImage_(const cv::Mat &input_img, cv::Mat &out_img) {
    if (input_img.empty()) {
        cout << "invalid input image, crop failed." << endl;
        return;
    }

    int index = -1;
    int count = 0;

    int threshold = input_img.rows / 20;
    for (int c = 0; c < input_img.cols; c += 10) {
        count = 0;
        for (int r = 0; r < input_img.rows; r += 10) {
            // random set
            if (input_img.at<short>(r, c) < 0 || input_img.at<short>(r, c) > SHRT_MAX / 2) {
                ++count;
                if (count >= threshold) {
                    index = c;
                    break;
                }
            }
        }
    }
    
    if (index == -1) {
        cout << "We don't need to crop anythings!" << endl;
        return;
    }
    input_img(cv::Rect(0, 0, index, input_img.rows)).copyTo(out_img);
    if (out_img.empty()) {
        out_img = input_img.clone();
    }
}

void ComputeHomography::Pixel2CamVec_(const std::vector<cv::KeyPoint> &kpts, std::vector<cv::Point3f> &p3d) {

    if (kpts.size() == 0) {
        cout << "empty keypoints container. return." << endl;
        return;
    }

    vector<cv::Point3f> p_cam;
    for (int i = 0; i < kpts.size(); ++i) {
        cv::Point3f p(
            (kpts[i].pt.x - mK_.at<float>(0, 2)) / mK_.at<float>(0, 0),
            (kpts[i].pt.y - mK_.at<float>(1, 2)) / mK_.at<float>(1, 1),
            1
        );
        p_cam.push_back(p);
    }

    p3d.clear();
    p3d.insert(p3d.end(), p_cam.begin(), p_cam.end());
}

void ComputeHomography::Pixel2Cam_(const cv::KeyPoint &kpts, cv::Point3f &p3d) {
    
    cv::Point3f p(
        (kpts.pt.x - mK_.at<float>(0, 2)) / mK_.at<float>(0, 0),
        (kpts.pt.y - mK_.at<float>(1, 2)) / mK_.at<float>(1, 1),
        1
    );
    p3d = p;
}