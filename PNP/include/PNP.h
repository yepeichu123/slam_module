#ifndef PNP_H
#define PNP_H

#include <opencv2/core/core.hpp>
#include <Eigen/Core>


enum SolvePnpType {
    PNP_CV,
    PNP_BA,
    PNP_COMB
};

class PNP {
    public:
        PNP(const cv::Mat& K);

        ~PNP();

        bool RunPNP(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point2f>& cur_p2d,
            cv::Mat& R, cv::Mat& t, const SolvePnpType& type = SolvePnpType::PNP_CV);

        float CheckPoseError(const std::vector<cv::Point3f>& ref_p3d, const std::vector<cv::Point2f>& cur_p2d,
            const cv::Mat& R, const cv::Mat& t);

    private:

        bool ComputeRelativePosePNP_(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point2f>& cur_p2d,
            cv::Mat& R, cv::Mat& t);

        bool ComputeRelativePosePNPByBA_(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point2f>& cur_p2d,
            cv::Mat& R, cv::Mat& t);

        void ChooseGoodMatching(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point2f>& cur_p2d,
            std::vector<char>& inliers);

        void Cam2Pixel_(const cv::Mat& p3d, cv::Mat& pixel);

        void Pixel2Cam_(const cv::Mat& pixel, cv::Mat& p3d);

        void ConvertPoseMatToEigen_(const cv::Mat& R, const cv::Mat &t, Eigen::Matrix3d& Rot, Eigen::Vector3d& trans);

        void ConvertPoseEigenToMat_(const Eigen::Matrix3d& Rot, const Eigen::Vector3d& trans, cv::Mat& R, cv::Mat& t);

        void ConvertPointMatToEigen_(const cv::Point3f& ref_p3d, const cv::Point2f& cur_p2d, Eigen::Vector3d& p3d, Eigen::Vector2d& p2d);

        cv::Mat m_K_;
        Eigen::Matrix3d m_eK_;

};

#endif // PNP_H