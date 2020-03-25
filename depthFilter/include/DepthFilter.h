#ifndef DEPTH_FILTER_H
#define DEPTH_FILTER_H

#include "Points.h"
#include <vector>

class Points;

class DepthFilter {
    public:
        DepthFilter(const cv::Mat &K);

        ~DepthFilter();

        // run depth filter 
        void RunSingleDepthFilter(Points &p_1, Points &p_2);

        // relative pose : [R | t] means from frame 2 to frame 1
        void ComputeTriangulatePoint(const cv::Mat &R, const cv::Mat &t, 
            Points &p_1, Points &p_2);

        // compute uncertainty
        void ComputeUncertainty(const cv::Mat &R, const cv::Mat &t, 
            Points &p_1, Points &p_2);

    private:

        cv::Mat mK_;

};

#endif // DEPTH_FILTER_H