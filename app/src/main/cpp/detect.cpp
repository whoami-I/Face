#include "opencv2/opencv.hpp"
#include "detect.h"

using namespace cv;
int min_face_size = 20;

extern "C"
void JNIFUNCF(Detector, nativeDetectFace, jobject bitmap,
              jint width, jint height) {
    void *destination = 0;
    int ret = 0;
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &destination)) < 0) {
        LOG("First Bitmap LockPixels Failed return=%d!", ret);
        return;
    }
    cv::Mat img(height, width, CV_8UC4, destination);
    cv::Mat im;
    int from_to[] = {0, 2, 1, 1, 2, 0};
    cv::mixChannels(&img, 1, &im, 1, from_to, 3);
    cv::resize(img, im, cv::Size(400.0f * width / height, 400));

    //begin pnet
    int net_size = 12;
    float scale_factor = 0.709;
    float current_scale = (float) net_size / min_face_size;

    cv::Mat norm_img = normalizeMat(im, current_scale);
    int currentH = norm_img.rows;
    int currentW = norm_img.cols;
    while (MIN(currentH, currentW) > net_size) {
        float *byte_buffer = 0;
        env->NewDirectByteBuffer(&byte_buffer,
                                 norm_img.rows * norm_img.cols * norm_img.depth());

        free(byte_buffer);
    }
    AndroidBitmap_unlockPixels(env, bitmap);
}

cv::Mat normalizeMat(cv::Mat in, float scale) {
    float h = in.rows * scale;
    float w = in.cols * scale;
    cv::Mat resizedImg;
    cv::resize(in, resizedImg, cv::Size(h, w), 0, 0, cv::InterpolationFlags::INTER_LINEAR);
    Mat norm_img;
    resizedImg.convertTo(norm_img, CV_32FC3);
    return norm_img;
}

