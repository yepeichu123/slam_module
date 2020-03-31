#include "ComputeHomography.h"

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
// c++
#include <iostream>
#include <sstream>
#define IMG_NUM 12

using namespace cv;
using namespace std;

void ReadImagesFromFile(const string &path, vector<Mat> &images);

void FeatureExtraction(const Mat &img, vector<KeyPoint> &kpts, Mat& desp);

void FeatureMatching(const Mat &desp_1, const Mat &desp_2, vector<DMatch> &matches);

int main(int argc, char** argv) {

    if (argc != 2) {
        cout << "Please input ./bin/ComputeHomography ./data/." << endl;
        return 1;
    }

    vector<Mat> images;
    ReadImagesFromFile(argv[1], images);
    if (images.size() < 9) {
        cout << "Read images failed, please check it again!" << endl;
        return 1;
    }
    Mat K = (Mat_<float>(3,3) << 518.0, 0, 325.5,
                                0, 519.0, 253.5,
                                0, 0, 1);

    Mat img_1 = images[0];
    Mat img_2 = images[8];
    vector<KeyPoint> kpts_1, kpts_2;
    Mat desp_1, desp_2;
    vector<DMatch> matches;
    FeatureExtraction(img_1, kpts_1, desp_1);
    FeatureExtraction(img_2, kpts_2, desp_2);
    FeatureMatching(desp_1, desp_2, matches);
    Mat out_img;
    drawMatches(img_1, kpts_1, img_2, kpts_2, matches, out_img);
    imshow("matching_img", out_img);
    waitKey(0);

    ComputeHomography homo(K);
    Mat H;
    homo.RunComputeHomography(kpts_1, kpts_2, matches, H);
    Mat stitch_img;
    homo.ImagesStitch(img_1, img_2, H, stitch_img);
    if (!stitch_img.empty()) {
        imshow("stitch image", stitch_img);
        waitKey(0);
    }
    return 0;
}

void FeatureExtraction(const Mat &img, vector<KeyPoint> &kpts, Mat& desp) {
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    sift->detectAndCompute(img, Mat(), kpts, desp);
}

void FeatureMatching(const Mat &desp_1, const Mat &desp_2, vector<DMatch> &matches) {

    matches.clear();
    
    vector<vector<DMatch>> temp_matches;
    Ptr<FlannBasedMatcher> flannMatcher = FlannBasedMatcher::create();
    std::vector<cv::Mat> train_dsc(1, desp_2);
    flannMatcher->add(train_dsc);
    flannMatcher->train();
    flannMatcher->knnMatch(desp_1, temp_matches, 2);

    for (auto m : temp_matches) {
        if (m[0].distance < 0.6*m[1].distance) {
            matches.push_back(m[0]);
        }
    }
    cout << "We get " << matches.size() << " good matching pairs!" << endl;
}

void ReadImagesFromFile(const string &path, vector<Mat> &images) {
    string img_path = path;
    for (int i = 0; i < IMG_NUM; ++i) {
        stringstream ss;
        ss << img_path << i << ".png";
        string img_file;
        ss >> img_file;

        Mat img = imread(img_file, IMREAD_COLOR);
        if (!img.empty()) {
            images.push_back(img);
        }
    }
    cout << "We read " << images.size() << " images from " << img_path << endl;
}