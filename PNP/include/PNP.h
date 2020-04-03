#ifndef PNP_H
#define PNP_H

#include <opencv2/core/core.hpp>

class PNP {
    public:
        PNP(const cv::Mat &K);

        ~PNP();

        bool RunPNP(std::vector<cv::Point3f> &ref_p3d, std::vector<cv::Point2f> &cur_p2d,
            cv::Mat &R, cv::Mat &t);

        float CheckPoseError(const std::vector<cv::Point3f> &ref_p3d, const std::vector<cv::Point2f> &cur_p2d,
            const cv::Mat &R, const cv::Mat &t);

    private:

        bool ComputeRelativePosePNP_(std::vector<cv::Point3f> &ref_p3d, std::vector<cv::Point2f> &cur_p2d,
            cv::Mat &R, cv::Mat &t);

        void ChooseGoodMatching(std::vector<cv::Point3f> &ref_p3d, std::vector<cv::Point2f> &cur_p2d,
            std::vector<char> &inliers);

        void Cam2Pixel_(const cv::Mat &p3d, cv::Mat &pixel);

        void Pixel2Cam_(const cv::Mat &pixel, cv::Mat &p3d);

        cv::Mat m_K_;

};

#endif // PNP_H