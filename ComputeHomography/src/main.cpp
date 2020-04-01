#include "ComputeHomography.h"

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
// c++
#include <iostream>
#include <sstream>
#define IMG_NUM 10

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

    ComputeHomography homo(K);
    Mat stitch_img_final;
    Mat ref_img = images[0];
    for (int i = 1; i < images.size(); ++i) {
        cout << "\n Image " << i << " : " << endl;

        Mat cur_img = images[i];
        vector<KeyPoint> kpts_1, kpts_2;
        Mat desp_1, desp_2;
        vector<DMatch> matches;
        FeatureExtraction(ref_img, kpts_1, desp_1);
        FeatureExtraction(cur_img, kpts_2, desp_2);
        FeatureMatching(desp_1, desp_2, matches);

        Mat H;
        homo.RunComputeHomography(kpts_1, kpts_2, matches, H);
        Mat stitch_img;
        if (!stitch_img_final.empty()) {
            stitch_img_final.copyTo(ref_img);
        }
        cout << "Stitch image!" << endl;

        homo.ImagesStitch(ref_img, cur_img, H, stitch_img);
        if (!stitch_img.empty()) {
            stitch_img.copyTo(stitch_img_final);
        }
        cur_img.copyTo(ref_img);
    }

    if (!stitch_img_final.empty()) {
        string out_file = string(argv[1]) + "stitched_img.png";
        imwrite(out_file, stitch_img_final);
        imshow("stitched_img", stitch_img_final);
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
            int padding = 10;
            Mat crop_img = img(Rect(padding, padding, img.cols-padding*2, img.rows-padding*2)).clone();
            images.push_back(crop_img);
        }
    }
    cout << "We read " << images.size() << " images from " << img_path << endl;
}