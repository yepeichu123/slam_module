#ifndef ICP_H
#define ICP_H
#include "ICP_G2O.h"
#include "ICP_G2O.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

enum ICP_TYPE {
    SVD,
    BA
};

class ICP {
    public:
        ICP();

        ~ICP();

        bool RunICP(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point3f>& cur_p3d,
            cv::Mat& R, cv::Mat &t, const ICP_TYPE& type = ICP_TYPE::SVD);

        bool SolveICPBySVD(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point3f>& cur_p3d,
            cv::Mat& R, cv::Mat &t);

        bool SovleICPByBA(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point3f>& cur_p3d,
            cv::Mat& R, cv::Mat &t);

        float CheckPoseError(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point3f>& cur_p3d,
            cv::Mat& R, cv::Mat &t);

    private:

        void ConvertP3dToMat_(const cv::Point3f& p3d, cv::Mat& mp3d);

        void ConvertP3dToMatVec_(const std::vector<cv::Point3f>& vp3d, std::vector<cv::Mat>& vmp3d);

        void ConvertPoseMatToEigen_(const cv::Mat& R, const cv::Mat& t, Eigen::Matrix3d& Rot, Eigen::Vector3d& trans);

        void ConvertPoseEigenToMat_(const Eigen::Matrix3d& Rot, const Eigen::Vector3d& trans, cv::Mat& R, cv::Mat& t);

        void ConvertPoint3fToEigen_(const cv::Point3f& p3d, Eigen::Vector3d& point3d);
};


#endif // ICP_H