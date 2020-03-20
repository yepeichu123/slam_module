#ifndef TRIANGULAR_POINTS_H
#define TRIANGULAR_POINTS_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

enum TYPE {
    ORBSLAM,
    VINS_MONO,
    CV,
    DEFAULT
};

class TriangularPoints {
    public:
        TriangularPoints(const cv::Mat &K);

        ~TriangularPoints();

        // proj_pose means P = K * T
        // !proj_pose means P = T, so we need to compute P = K * T
        void runTriangularPoints(const TYPE &type,
            const cv::Mat &Pose1, const cv::Mat &Pose2,
            std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2, 
            std::vector<cv::DMatch> &matches, std::vector<cv::Point3f> &points3d);

    private:

        // the method ORBSLAM used.
        void ORBSLAMTriangular_(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2, cv::Mat &X3D);

        // the method VINS-Mono used.
        void VINSTriangular_(Eigen::Vector2f &p2d_1, Eigen::Vector2f &p2d_2, Eigen::Vector3f &p3d);

        // the method OpenCV used 
        void CVTriangular_(std::vector<cv::DMatch> &matches,
            const std::vector<cv::KeyPoint> &kpt1, const std::vector<cv::KeyPoint> &kpt2, std::vector<cv::Point3f> &points3d);

        void DefaultTriangular_(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2, cv::Mat &X3D);

        void ConstructProjectMatrix_(const cv::Mat &Pose1, const cv::Mat &Pose2);

        void ConvertMatToEigen_();

        void ComputeAntisymmetricMatrix_(const cv::Mat &p, cv::Mat &matrix);

        // camera intrinsics
        float mffx_;
        float mffy_;
        float mfcx_;
        float mfcy_;

        // flags for exit program
        bool mbWrong_;

        // project matrix
        cv::Mat mmP1_;
        cv::Mat mmP2_;
        Eigen::Matrix<float, 3, 4> meP1_;
        Eigen::Matrix<float, 3, 4> meP2_;        

        // camera pose
        cv::Mat mmPose1_;
        cv::Mat mmPose2_;
};

#endif // TRIANGULAR_POINTS_H  