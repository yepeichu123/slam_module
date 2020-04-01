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

        float CheckPoseErrorWithK(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
            const std::vector<cv::DMatch> &matches, const cv::Mat &H);

        void ImagesStitch(const cv::Mat &ref_img, const cv::Mat &cur_img, const cv::Mat &H, cv::Mat &out_img);

        // with K means we should use normalized plane points 
        void CalcHomoMatWithK(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
            cv::Mat &H);

        // without K means we should use pixel plane points (original plane which OpenCV used)
        void CalcHomoMatWithoutK(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
            cv::Mat &H);

    private:

        void ComputeHomographyMat_(const std::vector<cv::KeyPoint> &ref_kpts, const std::vector<cv::KeyPoint> &cur_kpts,
            std::vector<cv::DMatch> &matches, cv::Mat &H);

        void CropEmptyImage_(const cv::Mat &input_img, cv::Mat &out_img);

        void Pixel2CamVec_(const std::vector<cv::KeyPoint> &kpts, std::vector<cv::Point3f> &p3d);

        void Pixel2Cam_(const cv::KeyPoint &kpts, cv::Point3f &p3d);

        cv::Mat mK_;
};

#endif // COMPUTE_HOMOGRAPHY_H