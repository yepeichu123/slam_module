#ifndef EPIPOLAR_CONSTRAIN_H
#define EPIPOLAR_CONSTRAIN_H

#include <opencv2/core/core.hpp>

class EpipolarConstrain {
    public:
        EpipolarConstrain(const cv::Mat &K);

        ~EpipolarConstrain();

        bool ComputeRelativePose(std::vector<cv::KeyPoint> &ref_kpts, std::vector<cv::KeyPoint> &cur_kpts, 
            std::vector<cv::DMatch> &matches, cv::Mat &R, cv::Mat &t);

        int IterateByRANSAC(std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &ref_kpt, std::vector<cv::KeyPoint> &cur_kpt,
            cv::Mat &R, cv::Mat &t);

        float CheckPoseError(std::vector<cv::Point2f> &ref_pixel, std::vector<cv::Point2f> &cur_pixel,
            cv::Mat &R, cv::Mat &t);

        void EpipolarConstrainFor8Points(std::vector<cv::Point2f> &ref_pixel, std::vector<cv::Point2f> &cur_pixel,
            cv::Mat &E);

        void FindBestPoseFromEssMat(const cv::Mat &E, std::vector<cv::Point2f> &ref_pixel, std::vector<cv::Point2f> &cur_pixel,
            cv::Mat &R, cv::Mat &t);

    private:

        // camera intrinsics
        cv::Mat mK_;
};

#endif // EPIPOLAR_CONSTRAIN_H