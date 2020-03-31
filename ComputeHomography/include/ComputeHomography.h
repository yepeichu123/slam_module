#ifndef COMPUTE_HOMOGRAPHY_H
#define COMPUTE_HOMOGRAPHY_H

#include <opencv2/core/core.hpp>

class ComputeHomography {
    public:
        ComputeHomography(const cv::Mat &K);

        ~ComputeHomography();

        void RunComputeHomography(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
            std::vector<cv::DMatch> &matches, cv::Mat &H);

        float CheckPoseError(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
            const std::vector<cv::DMatch> &matches, const cv::Mat &H);

        void ImagesStitch(const cv::Mat &ref_img, const cv::Mat &cur_img, const cv::Mat &H, cv::Mat &out_img);

    private:

        cv::Mat mK_;
};

#endif // COMPUTE_HOMOGRAPHY_H