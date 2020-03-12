#include <iostream>
#include <vector>
#include "FeatureExtraction.h"
#include "FeatureMatching.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    if (argc != 3) {
        cout << "Please go to the source dir, and input ./FeatureMatching ./data/1.png ./data/2.png." << endl;
        return 1;
    }

    Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img_2 = imread(argv[2], IMREAD_GRAYSCALE);

    // feature extracting
    string feat_type = "ORB";
    vector<KeyPoint> kpts_1, kpts_2;
    Mat desp_1, desp_2;
    Ptr<FeatureExtraction> feat_extra = new FeatureExtraction(feat_type, img_1);
    feat_extra->setDivedeImg_(2, 2);
    feat_extra->runFeatureExtractDiv(kpts_1, desp_1);
    // feat_extra->runFeatureExtractNoDiv(kpts_1, desp_1);
    feat_extra->SetImage(img_2);
    // feat_extra->runFeatureExtractNoDiv(kpts_2, desp_2);
    feat_extra->runFeatureExtractDiv(kpts_2, desp_2);

    // feature matching 
    string matching_type = "FLANN";
    // string matching_type = "BF";
    vector<DMatch> matches;
    Ptr<FeatureMatching> feat_matching = new FeatureMatching(matching_type, feat_type);
    feat_matching->RunFeatureMatching(desp_1, desp_2, matches);

    // draw matching
    Mat out_img;
    // for BF matching 
    // drawMatches(img_1, kpts_1, img_2, kpts_2, matches, out_img);
    // for FLANN matching 
    drawMatches(img_2, kpts_2, img_1, kpts_1, matches, out_img);
    if (!out_img.empty()) {
        string path = "./data/out_" + matching_type + "_" + feat_type + ".png";
        imwrite(path, out_img);
        imshow("matching_image", out_img);
        waitKey(0);
    }

    return 0;
}