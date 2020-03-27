// c++
#include <iostream>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    if (argc != 3) {
        cout << "Please input ./bin/stereoMatching ./data/left.png ./data/right.png." << endl;
        return 1;
    }

    // Read images 
    Mat left_img = imread(argv[1], IMREAD_GRAYSCALE);
    Mat right_img = imread(argv[2], IMREAD_GRAYSCALE);
    if (left_img.empty() || right_img.empty()) {
        cout << "Empty images! Please check your path!" << endl;
        return 1;
    }

    // camera intrinsics
    Mat K = (Mat_<float>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

    // Stereo Matching 
    

    return 0;
}