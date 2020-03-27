#include "StereoMatching.h"

#include <iostream>
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
}

StereoMatching::~StereoMatching() {

}

void StereoMatching::RunStereoMatching(const cv::Mat &left_img, const cv::Mat &right_img,
    const cv::Mat &R, const cv::Mat &t, cv::Mat &disparity_img) {

    if (left_img.empty() || right_img.empty()) {
        cout << "Input empty images!" << endl;
        return;
    }

    mLeftImg_ = left_img.clone();
    mRightImg_ = right_img.clone();

    mR_ = R.clone();
    mt_ = t.clone();

    // go through each pixel  
    for (int r = mnPadding_; r < mnRows_ - mnPadding_; ++r) {
        for (int c = mnPadding_; c < mnCols_ - mnPadding_; ++c) {
            
        }
    }
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

    if (best_scores > 0.9f) {
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
            left_avg += mLeftImg_.at<float>(j,i);

            i = right_pixel.x + col;
            j = right_pixel.y + row;
            right_avg += mRightImg_.at<float>(j,i);

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
            float left = mLeftImg_.at<float>(j,i);

            i = right_pixel.x + col;
            j = right_pixel.y + row;
            float right = mRightImg_.at<float>(j,i);

            left_x_right += (left - left_avg) * (right - right_avg);
            left_square += (left - left_avg) * (left - left_avg);
            right_square += (right - right_avg) * (right - right_avg);
        }
    }

    float ncc = left_x_right / sqrt(left_square * right_square);

    if (ncc > 1 || ncc < 0) {
        return 0.f;
    }
    return ncc;

}