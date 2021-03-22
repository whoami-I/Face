#include "opencv2/opencv.hpp"
#include "detect.h"
#include "inference.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

using namespace cv;
using namespace std;

std::unique_ptr<InferenceContext> ctx;

extern "C"
void JNIFUNCF(Detector, nativeInit, jobject assetManager) {
    if (ctx == NULL) {
        ctx = std::unique_ptr<InferenceContext>(new InferenceContext());
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        AAsset *pmodel_asset = AAssetManager_open(mgr, "pmodel.tflite",
                                                  AASSET_MODE_STREAMING);
        int64_t pmodel_buffer_length = AAsset_getLength(pmodel_asset);
        char *pmodel_buffer = new char[pmodel_buffer_length];
        AAsset_read(pmodel_asset, pmodel_buffer, pmodel_buffer_length);

        AAsset *rmodel_asset = AAssetManager_open(mgr, "rmodel.tflite",
                                                  AASSET_MODE_STREAMING);
        int64_t rmodel_buffer_length = AAsset_getLength(rmodel_asset);
        char *rmodel_buffer = new char[rmodel_buffer_length];
        AAsset_read(rmodel_asset, rmodel_buffer, rmodel_buffer_length);

        AAsset *omodel_asset = AAssetManager_open(mgr, "omodel.tflite",
                                                  AASSET_MODE_STREAMING);
        int64_t omodel_buffer_length = AAsset_getLength(omodel_asset);
        char *omodel_buffer = new char[omodel_buffer_length];
        AAsset_read(omodel_asset, omodel_buffer, omodel_buffer_length);

        ctx->initModel(pmodel_buffer, pmodel_buffer_length, rmodel_buffer,
                       rmodel_buffer_length, omodel_buffer, omodel_buffer_length);

        AAsset_close(pmodel_asset);
        AAsset_close(rmodel_asset);
        AAsset_close(omodel_asset);
        delete[]pmodel_buffer;
        delete[]rmodel_buffer;
        delete[]omodel_buffer;
    }
}


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
    cv::Mat im(height, width, CV_8UC3);
    int from_to[] = {0, 2, 1, 1, 2, 0};
    cv::mixChannels(&img, 1, &im, 1, from_to, 3);
    cv::resize(img, im, cv::Size(400.0f * width / height, 400));
    vector<ScoreBox> boxes;
    ctx->detectNet(im, boxes);


    AndroidBitmap_unlockPixels(env, bitmap);
}

cv::Mat normalizeMat(cv::Mat in, float scale) {
    float h = in.rows * scale;
    float w = in.cols * scale;
    cv::Mat resizedImg;
    cv::resize(in, resizedImg, cv::Size(w, h), 0, 0, cv::InterpolationFlags::INTER_LINEAR);
    Mat norm_img;
    resizedImg.convertTo(norm_img, CV_32FC3);
//    norm_img -= 127.5f;
//    norm_img /= 128.0f;
    norm_img = (norm_img - 127.5f) / 128;
    return norm_img;
}

