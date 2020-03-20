#ifndef EPIPOLAR_SEARCH_H
#define EPIPOLAR_SEARCH_H

#include <opencv2/core/core.hpp>

class EpipolarSearch {
    public:
        EpipolarSearch(const cv::Mat &K);

        EpipolarSearch(const cv::Mat &K, const cv::Mat &ref_img, const cv::Mat &cur_img);

        ~EpipolarSearch();

        // R and t means transformation from reference frame to current frame
        bool RunEpipolarSearch(const cv::Mat &ref_p, cv::Mat &cur_p,
            const cv::Mat &R, cv::Mat &t);

        // R and t means transformation from reference frame to current frame
        bool RunEpipolarSearch(const cv::Mat &ref_img, const cv::Mat &cur_img, 
            const cv::Mat &ref_p, cv::Mat &cur_p,
            const cv::Mat &R, cv::Mat &t);

        void SetupImage(const cv::Mat &ref_img, const cv::Mat &cur_img);

    private:
        // from pixel to normalized plane of camera frame 
        void PixelToCam_(const cv::Mat &pixel, cv::Mat &p3d);

        // from normalized plane of camera frame to pixel
        void CamToPixel_(const cv::Mat &p3d, cv::Mat &pixel);

        // check if the pixel is inside the image 
        bool isInliers_(const cv::Mat &pixel);

        // compute NCC scores
        float ComputeNCCScores_(const cv::Mat &ref_p, const cv::Mat &cur_p); 

        // current image and reference image 
        cv::Mat mmRefImg_;
        cv::Mat mmCurImg_;

        // camera intrisics
        cv::Mat mmK_;

        // NCC window
        int mnWin_;
};

#endif // EPIPOLAR_SEARCH_H