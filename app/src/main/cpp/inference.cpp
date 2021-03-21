#include "inference.h"
#include "logutils.h"

#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include <include/detect.h>

using namespace cv;

InferenceContext::InferenceContext() : backend() {}

int InferenceContext::get_num(JNIEnv *env, jobject bitmap, int width, int height) {
//    void *destination = 0;
//    int ret = 0;
//    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &destination)) < 0) {
//        LOG("First Bitmap LockPixels Failed return=%d!", ret);
//        return -1;
//    }
//
//    Mat src(height, width, CV_8UC4, destination);
//    Mat out(height, width, CV_8UC1);
//    float *inputArray = backend.getInputPtr();
//    Mat input(height, width, CV_32FC1, inputArray);
//    int fromTo[] = {0, 0};
//    mixChannels(&src, 1, &out, 1, fromTo, 1);
//    out.convertTo(input, CV_32FC1);
//    backend.invoke();
//    float *result = backend.getOutputPtr();
//
//    float max = -100000;
//    int index = -1;
//    for (int i = 0; i < 10; ++i) {
//        if (max < result[i]) {
//            max = result[i];
//            index = i;
//        }
////        LOG("the result is %f\n", result[i]);
//    }
//
//    AndroidBitmap_unlockPixels(env, bitmap);
    return 1;
}

void InferenceContext::initModel(char *pmodel_buffer, size_t pmodel_legth,
                                 char *rmodel_buffer, size_t rmodel_legth,
                                 char *omodel_buffer, size_t omodel_legth) {
    backend.loadPmodel(pmodel_buffer, pmodel_legth);
    backend.loadRmodel(rmodel_buffer, rmodel_legth);
    backend.loadOmodel(omodel_buffer, omodel_legth);
}

/**
 * @param img bgr排列的图片
 */
void InferenceContext::detectPNet(Mat img) {
    //begin pnet
    int net_size = 12;
    float scale_factor = 0.709;
    float current_scale = (float) net_size / min_face_size;

    cv::Mat norm_img = normalizeMat(img, current_scale);
    int currentH = norm_img.rows;
    int currentW = norm_img.cols;

    while (MIN(currentH, currentW) > net_size) {
        float *input = backend.pmodel_interpreter->typed_input_tensor<float>(0);

        backend.pmodel_interpreter->Invoke();
    }
}

