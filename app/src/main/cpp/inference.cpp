#include "inference.h"
#include "logutils.h"

#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include <include/detect.h>


using namespace cv;
using namespace std;

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
void InferenceContext::detectPNet(Mat &img, vector<ScoreBox> &boxlist) {
    //begin pnet
    int net_size = 12;
    float scale_factor = 0.709;
    float current_scale = (float) net_size / min_face_size;

    cv::Mat norm_img = normalizeMat(img, current_scale);
    int currentH = norm_img.rows;
    int currentW = norm_img.cols;
    vector<ScoreBox> validBox;
    while (MIN(currentH, currentW) > net_size) {
        std::vector<int> dims;
        dims.push_back(1);
        dims.push_back(norm_img.rows);
        dims.push_back(norm_img.cols);
        dims.push_back(3);
        backend.pmodel_interpreter->ResizeInputTensor(0, dims);
        if (backend.pmodel_interpreter->AllocateTensors() != kTfLiteOk) {
            throw std::runtime_error("Could not allocate tensors");
        }
        float *input = backend.pmodel_interpreter->typed_input_tensor<float>(0);
        Mat inputImg(norm_img.rows, norm_img.cols, CV_32FC3, input);
        norm_img.copyTo(inputImg);

        backend.pmodel_interpreter->Invoke();
        std::vector<int> v1 = backend.pmodel_interpreter->outputs();
        float *boxOutput = backend.pmodel_interpreter->typed_output_tensor<float>(0);
        TfLiteTensor *box = backend.pmodel_interpreter->tensor(v1.at(0));
        TfLiteIntArray *boxArray = box->dims;
        Mat boxMat(boxArray->data[1], boxArray->data[2], CV_32FC4, boxOutput);

        float *classifyOutput = backend.pmodel_interpreter->typed_output_tensor<float>(1);
        //可能有问题v2
        std::vector<int> v2 = backend.pmodel_interpreter->outputs();
        TfLiteTensor *classify2 = backend.pmodel_interpreter->tensor(v1.at(1));
        TfLiteIntArray *classifyArray2 = classify2->dims;
        Mat classifyMat(classifyArray2->data[1],
                        classifyArray2->data[2], CV_32FC2, classifyOutput);
        vector<ScoreBox> tmpBox;
        getBoundingBox(classifyMat, boxMat, current_scale, 0.6f, tmpBox);

        current_scale *= scale_factor;
        norm_img = normalizeMat(img, current_scale);
        currentH = norm_img.rows;
        currentW = norm_img.cols;

        if (tmpBox.size() == 0) continue;
        //nms
        nms(tmpBox, 0.5f);
        validBox.insert(validBox.end(), tmpBox.begin(), tmpBox.end());
    }
    nms(validBox, 0.5f);
    float bw = 0;
    float bh = 0;
    for (std::vector<ScoreBox>::iterator it = validBox.begin();
         it != validBox.end(); ++it) {
        bw = it->x2 - it->x1;
        bh = it->y2 - it->y1;
        int x1 = it->x1 + bw * it->x1_shift;
        int y1 = it->y1 + bh * it->y1_shift;

        int x2 = it->x2 + bw * it->x2_shift;
        int y2 = it->y2 + bh * it->y2_shift;
        it->set(x1, y1, x2, y2);
    }
    for(auto it=validBox.begin();it!=validBox.end();it++){
        boxlist.push_back(*it);
    }
}

void InferenceContext::detectRNet(Mat &img, vector<ScoreBox> &boxlist, vector<ScoreBox> &pnetBox) {
    makeBoxSquare(pnetBox);
    vector<ScoreBox> tmpList;
    for (auto it = pnetBox.begin(); it != pnetBox.end(); it++) {
        int x1, y1, x2, y2, rectx1, recty1, rectx2, recty2;
        x1 = rectx1 = it->x1;
        y1 = recty1 = it->y1;
        x2 = rectx2 = it->x2;
        y2 = recty2 = it->y2;
        int h = img.rows;
        int w = img.cols;
        int top, bottom, right, left;
        top = bottom = right = left = 0;
        if (x1 < 0) {
            rectx1 = 0;
            left = -x1;
        }
        if (x2 >= w) {
            recty1 = w - 1;
            right = x2 - w + 1;
        }
        if (y1 < 0) {
            rectx2 = 0;
            top = -y1;
        }
        if (y2 >= h) {
            recty2 = h - 1;
            bottom = y2 - h + 1;
        }
        Rect rect(rectx1, recty1, rectx2 - rectx1, recty2 - recty1);
        Mat im = img(rect).clone();
        Mat out;
        if (rect.width != rect.height) {
            copyMakeBorder(im, out, top, bottom, left, right,
                           BORDER_CONSTANT, Scalar(0, 0, 0));
        } else {
            out = im;
        }
        cvtColor(out, out, CV_32FC3);

        float *input = backend.rmodel_interpreter->typed_input_tensor<float>(0);

        Mat r(24, 24, CV_32FC3, input);
        resize(out, r, Size(24, 24));
        r -= 127.5f;
        r /= 128.0f;
        backend.rmodel_interpreter->Invoke();
        float *box = backend.rmodel_interpreter->typed_tensor<float>(1);
        float *classify = backend.rmodel_interpreter->typed_tensor<float>(1);
        it->x1_shift = box[0];
        it->y1_shift = box[1];
        it->x2_shift = box[2];
        it->y2_shift = box[3];
        it->score = classify[1];
        if (classify[1] > 0.8f) {
            tmpList.push_back(*it);
        }
    }
    if (!tmpList.empty()) {
        nms(tmpList, 0.3f);
    }
    refineBox(boxlist, img.rows, img.cols);

    for(auto it=tmpList.begin();it!=tmpList.end();it++){
        boxlist.push_back(*it);
    }
}


void InferenceContext::detectONet(Mat &img, vector<ScoreBox> &boxlist, vector<ScoreBox> &rnetBox) {
    makeBoxSquare(rnetBox);
    vector<ScoreBox> tmpList;
    for (auto it = rnetBox.begin(); it != rnetBox.end(); it++) {
        int x1, y1, x2, y2, rectx1, recty1, rectx2, recty2;
        x1 = rectx1 = it->x1;
        y1 = recty1 = it->y1;
        x2 = rectx2 = it->x2;
        y2 = recty2 = it->y2;
        int h = img.rows;
        int w = img.cols;
        int top, bottom, right, left;
        top = bottom = right = left = 0;
        if (x1 < 0) {
            rectx1 = 0;
            left = -x1;
        }
        if (x2 >= w) {
            recty1 = w - 1;
            right = x2 - w + 1;
        }
        if (y1 < 0) {
            rectx2 = 0;
            top = -y1;
        }
        if (y2 >= h) {
            recty2 = h - 1;
            bottom = y2 - h + 1;
        }
        Rect rect(rectx1, recty1, rectx2 - rectx1, recty2 - recty1);
        Mat im = img(rect).clone();
        Mat out;
        if (rect.width != rect.height) {
            copyMakeBorder(im, out, top, bottom, left, right,
                           BORDER_CONSTANT, Scalar(0, 0, 0));
        } else {
            out = im;
        }
        cvtColor(out, out, CV_32FC3);

        float *input = backend.omodel_interpreter->typed_input_tensor<float>(0);

        Mat r(48, 48, CV_32FC3, input);
        resize(out, r, Size(48, 48));
        r -= 127.5f;
        r /= 128.0f;
        backend.omodel_interpreter->Invoke();
        float *box = backend.omodel_interpreter->typed_tensor<float>(1);
        float *classify = backend.omodel_interpreter->typed_tensor<float>(1);
        it->x1_shift = box[0];
        it->y1_shift = box[1];
        it->x2_shift = box[2];
        it->y2_shift = box[3];
        it->score = classify[1];
        if (classify[1] > 0.8f) {
            tmpList.push_back(*it);
        }
    }
    if (!tmpList.empty()) {
        nms(tmpList, 0.5f);
    }
    refineBox(boxlist, img.rows, img.cols);
    for(auto it=tmpList.begin();it!=tmpList.end();it++){
        boxlist.push_back(*it);
    }
}


void InferenceContext::getBoundingBox(Mat &scoreMat, Mat &boxMat, float scale, float threshold,
                                      std::vector<ScoreBox> &validBox) {
    const int stride = 2;
    const int cellsize = 12;
    Mat mat[2];
    split(scoreMat, mat);
    Mat score = mat[1];

    for (int i = 0; i < score.rows; ++i) {
        for (int j = 0; j < score.cols; ++j) {
            float s = score.at<float>(i, j);
            if (s > threshold) {
                ScoreBox scoreBox;
                scoreBox.x1 = round((stride * j + 1) / scale);
                scoreBox.y1 = round((stride * i + 1) / scale);
                scoreBox.x2 = round((stride * j + 1 + cellsize) / scale);
                scoreBox.y2 = round((stride * i + 1 + cellsize) / scale);
                float *p = boxMat.ptr<float>(i, j);
                scoreBox.x1_shift = *p;
                scoreBox.y1_shift = *(p + 1);
                scoreBox.x2_shift = *(p + 2);
                scoreBox.y2_shift = *(p + 3);
                scoreBox.score = s;
                validBox.push_back(scoreBox);
            }
        }
    }
}

bool cmpScore(ScoreBox lsh, ScoreBox rsh) {
    if (lsh.score < rsh.score)
        return true;
    else
        return false;
}

void InferenceContext::nms(std::vector<ScoreBox> &boundingBox_, const float overlap_threshold) {
    if (boundingBox_.empty()) {
        return;
    }
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    std::vector<int> vPick;
    int nPick = 0;
    std::multimap<float, int> vScores;
    const int num_boxes = boundingBox_.size();
    vPick.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i) {
        vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
    }
    while (vScores.size() > 0) {
        int last = vScores.rbegin()->second;
        vPick[nPick] = last;
        nPick += 1;
        for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
            int it_idx = it->second;
            maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
            //maxX1 and maxY1 reuse
            maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
            maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
            //IOU reuse for the area of two bbox
            IOU = maxX * maxY;
            IOU = IOU / (boundingBox_.at(it_idx).area() + boundingBox_.at(last).area() - IOU);
            if (IOU > overlap_threshold) {
                it = vScores.erase(it);
            } else {
                it++;
            }
        }
    }

    vPick.resize(nPick);
    std::vector<ScoreBox> tmp_;
    tmp_.resize(nPick);
    for (int i = 0; i < nPick; i++) {
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}

void InferenceContext::detectNet(Mat &img, vector<ScoreBox> &boxes) {
    //begin pnet
    vector<ScoreBox> tmpPBoxes;
    vector<ScoreBox> tmpRBoxes;
    detectPNet(img, tmpPBoxes);
    detectRNet(img, tmpRBoxes, tmpPBoxes);
    detectONet(img, boxes, tmpRBoxes);
}

void InferenceContext::refineBox(std::vector<ScoreBox> &vector, int rows, int cols) {
    for (auto it = vector.begin(); it != vector.end(); it++) {
        int bw, bh;
        bw = it->x2 - it->x1;
        bh = it->y2 - it->y1;
        int x1 = it->x1 + bw * it->x1_shift;
        int y1 = it->y1 + bh * it->y1_shift;

        int x2 = it->x2 + bw * it->x2_shift;
        int y2 = it->y2 + bh * it->y2_shift;
        it->set(x1, y1, x2, y2);
    }
}

void InferenceContext::makeBoxSquare(std::vector<ScoreBox> &boxs) {
    for (auto it = boxs.begin(); it != boxs.end(); it++) {
        int x1 = it->x1;
        int y1 = it->y1;
        int x2 = it->x2;
        int y2 = it->y2;
        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;
        if (w == h) continue;
        int maxSide = (h > w) ? h : w;
        x1 = x1 + w * 0.5 - maxSide * 0.5;
        y1 = y1 + h * 0.5 - maxSide * 0.5;
        it->x1 = x1;
        it->y1 = y1;
        it->x2 = x1 + maxSide;
        it->y2 = y1 + maxSide;
    }
}
