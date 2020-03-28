#ifndef STEREO_MATCHING_H
#define STEREO_MATCHING_H

#include <opencv2/core/core.hpp>

class StereoMatching {
    public:
        StereoMatching(const cv::Mat &K, const int &img_rows, const int &img_cols);

        ~StereoMatching();
    
        void RunStereoMatching(const cv::Mat &left_img, const cv::Mat &right_img,
            const cv::Mat &R, const cv::Mat &t, 
            cv::Mat &disparity_img, cv::Mat &depth_img);

    private:

        // epipolar search 
        bool EpipolarSearch_(cv::Point2f &left_pixel, cv::Point2f &right_pixel);

        // compute NCC scores
        float ComputeNCCScores_(cv::Point2f &left_pixel, cv::Point2f &right_pixel);

        // Bilinear interpolation
        float BilinearInterpolation();

        // camera intrinsics 
        cv::Mat mK_;

        // fixed transformation
        cv::Mat mR_, mt_;

        // camera matrix 
        cv::Mat mLeftImg_, mRightImg_;

        // image size 
        int mnRows_, mnCols_;

        // padding 
        int mnPadding_;

        // NCC windows 
        int mnNCCRows_, mnNCCCols_;
};

#endif // STEREO_MATCHING_H