#ifndef EPIPOLAR_CONSTRAIN_H
#define EPIPOLAR_CONSTRAIN_H

#include <opencv2/core/core.hpp>

class EpipolarConstrain {
    public:
        EpipolarConstrain(const cv::Mat &K);

        ~EpipolarConstrain();

        bool ComputeRelativePose();

        int MatchingByRANSAC();

        float CheckPoseError();

    private:

        // camera intrinsics
        cv::Mat mK_;


};

#endif // EPIPOLAR_CONSTRAIN_H