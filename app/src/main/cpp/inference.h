#ifndef INFERENCE_H
#define INFERENCE_H

#include "utils.h"
#include "tflite.h"
#include <jni.h>
using namespace std;

class ScoreBox {
public:
    int x1;
    int x2;
    int y1;
    int y2;
    float x1_shift;
    float x2_shift;
    float y1_shift;
    float y2_shift;

    float score;

    float area() {
        float a = (x2 - x1 + 1) * (y2 - y1 + 1);
        return a > 0 ? a : 0;
    }

    void set(int xx1, int yy1, int xx2, int yy2) {
        x1 = xx1;
        x2 = xx2;
        y1 = yy1;
        y2 = yy2;
    }
};

class InferenceContext {
private:
    Dims inputDims{513, 513,
                   3}; //512 256 + 1 TODO(Paweł Subko) hardcoded model property -> move to cfg
    Dims outputDims{129, 129,
                    2}; // 128 64 + 1 TODO(Paweł Subko) hardcoded model property -> move to cfg

    // Computed masks
    cv::Mat nativeSizeMask;
    cv::Mat currentMask;

    // Tmp mats
    cv::Mat intermediateSizeImgRGB;
    TfBackend backend;
    int min_face_size = 20;

public:

    InferenceContext();

    ~InferenceContext() {};

//    void computeSegmentationMask(cv::Mat &inputMat, bool CRF);
//    void applyEffect(cv::Mat &dstMat, cv::Mat &effectMat, bool transferColor=true);
    int get_num(JNIEnv *env, jobject bitmap, int width, int height);

    void
    initModel(char *pmodel_buffer, size_t pmodel_legth, char *rmodel_buffer, size_t rmodel_legth,
              char *omodel_buffer, size_t omodel_legth);

    void detectNet(cv::Mat &img, vector <ScoreBox> &boxes);

    void getBoundingBox(cv::Mat &scoreMat, cv::Mat &boxMat, float scale, float threshold,
                        std::vector<ScoreBox> &validBox);

    void nms(std::vector<ScoreBox> &boundingBox_, const float overlap_threshold);

    void refineBox(std::vector<ScoreBox> &vector, int rows, int cols);

    void detectONet(cv::Mat &img, std::vector<ScoreBox> &boxlist, std::vector<ScoreBox> &rnetBox);

    void detectRNet(cv::Mat &img, std::vector<ScoreBox> &boxlist, std::vector<ScoreBox> &pnetBox);

    void detectPNet(cv::Mat &img, std::vector<ScoreBox> &boxlist);

    void makeBoxSquare(std::vector<ScoreBox> &vector);
};


#endif