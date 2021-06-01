#include "net.h"

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>

struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};


class NcnnFaceDetector {

    protected:

        inline float intersection_area(const FaceObject& a, const FaceObject& b);

        void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right);

        void qsort_descent_inplace(std::vector<FaceObject>& faceobjects);

        void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold);

        ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales);

        void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, float prob_threshold, std::vector<FaceObject>& faceobjects);

    public:
        NcnnFaceDetector();

        ncnn::Net retinaface;

        int detect_retinaface(const cv::Mat& bgr, std::vector<FaceObject>& faceobjects);

};