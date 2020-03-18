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
    CV
};

class TriangularPoints {
    public:
        TriangularPoints(const cv::Mat &K);

        ~TriangularPoints();

        void runTriangularPoints(const TYPE &type,
            const cv::Mat &Pose1, const cv::Mat &Pose2,
            std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2, 
            std::vector<cv::DMatch> &matches, std::vector<float> &depth);

    private:

        // the method ORBSLAM used.
        void ORBSLAMTriangular(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2, 
            const cv::Mat &P1, const cv::Mat &P2, const cv::Mat &X3D);

        // the method VINS-Mono used.
        void VINSTriangular(Eigen::Matrix<float, 3, 4> &Pose1, Eigen::Matrix<float, 3, 4> &Pose2,
            Eigen::Vector2f &p2d_1, Eigen::Vector2f &p2d_2, Eigen::Vector3d &p3d);

        // the method OpenCV used 
        void CVTriangular(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2, 
            const cv::Mat &P1, const cv::Mat &P2, const cv::Mat &X3D);

        // camera intrinsics
        float mdfx_;
        float mdfy_;
        float mdcx_;
        float mdcy_;

        // flags for exit program
        bool mbWrong_;
};

#endif // TRIANGULAR_POINTS_H  