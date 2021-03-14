#include "opencv2/opencv.hpp"
#include "filters.h"

extern "C"
void JNIFUNCF(AlgManager, nativeGrayOpencv, jobject bitmap,
              jint width, jint height) {
    void *destination = 0;
//    AndroidBitmap_lockPixels(env, bitmap, &destination);
    int ret = 0;
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &destination)) < 0) {
        LOG("First Bitmap LockPixels Failed return=%d!", ret);
        return;
    }
    cv::Mat src(height, width, CV_8UC4, destination);
    LOG("---------nativeGrayOpencv");
    //将图像转为灰度图
    cv::Mat grayImage;
    cv::cvtColor(src, grayImage, cv::COLOR_BGRA2GRAY);
    cv::cvtColor(grayImage, src, cv::COLOR_GRAY2BGRA);
    AndroidBitmap_unlockPixels(env, bitmap);
}

