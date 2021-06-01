#include <iostream>
#include <cstdio>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <chrono> 

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include "FaceDetector.h"

using namespace cv;
using namespace dnn;
using namespace std;
using namespace std::chrono; 


int main(int argc, char **argv) {
    std::cout << "Hello from main.." << std::endl;

    string image_file;
    if  (argc == 1)
    {
        image_file = "../img/human.jpg";
    }
    else if (argc == 2)
    {
        image_file = argv[1];
    }
    std::cout << "Processing " << image_file << std::endl;

    //// 2
    /// flag value 1 to imread means -- cv2.IMREAD_COLOR:
    /// It specifies to load a color image. 
    /// Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.
    cv::Mat frame = cv::imread(image_file.c_str(), 1);

    if (frame.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_file.c_str());
        return -1;
    }

    // //// 3
    std::vector<FaceObject> faceobjects;
    NcnnFaceDetector face_detector;
    face_detector.detect_retinaface(frame, faceobjects);

    std::cout << "faceobjects[0].rect.x: " << faceobjects[0].rect.x << std::endl;
    std::cout << "faceobjects[0].rect.y: " << faceobjects[0].rect.y << std::endl;
    std::cout << "faceobjects[0].rect.width: " << faceobjects[0].rect.width << std::endl;
    std::cout << "faceobjects[0].rect.height: " << faceobjects[0].rect.height << std::endl;

}