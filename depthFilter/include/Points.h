#ifndef POINTS_H
#define POINTS_H

#include <opencv2/core/core.hpp>

class Points{
    public:
        Points(const int &id);

        Points(const int &id, const cv::Mat &K, const cv::KeyPoint &kpts, const cv::Mat &desp, const float &depth);

        ~Points();

        void SetPointDepth(const float &depth);

        void SetKeyPoint(const cv::KeyPoint &kpts);

        void SetDescriptor(const cv::Mat &desp);

        void SetCameraIntrinsics(const cv::Mat &K);

        void SetDepthUncertainty(const float &depth_un);

        void UpdatePointDepth(const float &new_depth, const float &new_uncertainty);

        int GetPointId();

        cv::KeyPoint& GetKeyPoint();

        cv::Mat& GetDescriptor();

        float& GetDepth();

        float GetUncertainty();

    private:
    
        int mnId_;
        
        cv::Mat mK_;

        cv::KeyPoint mKpt_;
        
        cv::Mat mDesp_;

        float mDepth_;

        float mDepthUncertain_;
};

#endif // POINTS_H