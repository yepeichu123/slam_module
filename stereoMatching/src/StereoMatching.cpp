#include "StereoMatching.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace std;

StereoMatching::StereoMatching(const cv::Mat &K, const int &img_rows, const int &img_cols) {
    mK_ = K.clone();
    mnRows_ = img_rows;
    mnCols_ = img_cols;

    mnNCCRows_ = mnRows_ > mnCols_ ? (mnRows_/mnCols_ + 3) : 3;
    mnNCCRows_ /= 2;
    mnNCCCols_ = mnRows_ < mnCols_ ? (mnCols_/mnRows_ + 3) : 3;
    mnNCCCols_ /= 2;

    mnPadding_ = mnNCCRows_ > mnNCCCols_ ? (mnNCCRows_ + 2) : (mnNCCCols_ + 2);
    cout << "Setup default configures:" << endl;
    cout << "image rows = " << mnRows_ << ", image cols = " << mnCols_ << endl;
    cout << "NCC window rows = " << mnNCCRows_ << ", cols = " << mnNCCCols_ << endl;
}

StereoMatching::~StereoMatching() {

}

void StereoMatching::RunStereoMatching(const cv::Mat &left_img, const cv::Mat &right_img,
    const cv::Mat &R, const cv::Mat &t, 
    cv::Mat &disparity_img, cv::Mat &depth_img) {

    if (left_img.empty() || right_img.empty()) {
        cout << "Input empty images!" << endl;
        return;
    }

    if (left_img.channels() != 1) {
        cv::cvtColor(left_img, left_img, cv::COLOR_BGR2GRAY);
    }
    if (right_img.channels() != 1) {
        cv::cvtColor(right_img, right_img, cv::COLOR_BGR2GRAY);
    }

    mLeftImg_ = left_img.clone();
    mRightImg_ = right_img.clone();

    mR_ = R.clone();
    mt_ = t.clone();

    disparity_img = cv::Mat(mnRows_, mnCols_, 0);
    depth_img = cv::Mat(mnRows_, mnCols_, 0);
    float base_line = cv::norm(t);
    float f = mK_.at<float>(0,0);

    // go through each pixel  
    for (int r = mnPadding_; r < mnRows_ - mnPadding_; ++r) {
        int  count = 0;
        for (int c = mnPadding_; c < mnCols_ - mnPadding_; ++c) {
            cv::Point2f left_p(c, r);
            cv::Point2f right_p;
            if (!EpipolarSearch_(left_p, right_p)) {
                continue;
            }

            float d = right_p.x - left_p.x;
            disparity_img.at<uchar>(r,c) = d;
            depth_img.at<uchar>(r,c) = base_line * f / d;
            ++count;
        }
        cout << r << " : we get " << count << " matching pairs." << endl;
    } 
    /*
    for (int r = mnPadding_; r < disparity_img.rows - mnPadding_; ++r) {
        for (int c = mnPadding_; c < disparity_img.cols - mnPadding_; ++c) {
            float d = disparity_img.at<uchar>(r, c);
            int width = disparity_img.cols;
            if (d == 0) {
                d = (disparity_img.at<uchar>(r, c+1) + 
                     disparity_img.at<uchar>(r, c-1) + 
                     disparity_img.at<uchar>(r+1, c) + 
                     disparity_img.at<uchar>(r-1, c)
                ) / 4;
                disparity_img.at<uchar>(r, c) = d;
            }
        }
    }
    */
}

// epipolar search 
bool StereoMatching::EpipolarSearch_(cv::Point2f &left_pixel, cv::Point2f &right_pixel) {
    
    // go through all pixels in the epipolar line 
    float best_scores = 0;
    for (int i = mnPadding_; i < mnCols_- mnPadding_; ++i) {
        cv::Point2f temp_point(i, (int)left_pixel.y);
        float ncc = ComputeNCCScores_(left_pixel, temp_point);

        if (ncc > best_scores) {
            best_scores = ncc;
            right_pixel = temp_point;
        }
    }   

    if (best_scores > 0.95f) {
        return true;
    }
    return false;
}

// compute NCC scores
float StereoMatching::ComputeNCCScores_(cv::Point2f &left_pixel, cv::Point2f &right_pixel) {

    // NCC windows 
    float left_avg = 0, right_avg = 0;
    int count = 0;
    for (int row = -mnNCCRows_; row < mnNCCRows_; ++row) {
        for (int col = -mnNCCCols_; col < mnNCCCols_; ++col) {
            int i = left_pixel.x + col;
            int j = left_pixel.y + row;
            left_avg += mLeftImg_.at<uchar>(j,i);

            i = right_pixel.x + col;
            j = right_pixel.y + row;
            right_avg += mRightImg_.at<uchar>(j,i);

            ++count;
        }
    }
    left_avg /= count;
    right_avg /= count;

    float left_square = 0, right_square = 0;
    float left_x_right = 0;
    for (int row = -mnNCCRows_; row < mnNCCRows_; ++row) {
        for (int col = -mnNCCCols_; col < mnNCCCols_; ++col) {
            int i = left_pixel.x + col;
            int j = left_pixel.y + row;
            float left = mLeftImg_.at<uchar>(j,i);

            i = right_pixel.x + col;
            j = right_pixel.y + row;
            float right = mRightImg_.at<uchar>(j,i);

            left_x_right += (left - left_avg) * (right - right_avg);
            left_square += (left - left_avg) * (left - left_avg);
            right_square += (right - right_avg) * (right - right_avg);
        }
    }
    left_x_right /= count;
    left_square /= count;
    right_square /= count;

    float ncc = left_x_right / sqrt(left_square * right_square);

    if (ncc > 1 || ncc < 0) {
        return 0.f;
    }
    return ncc;

}