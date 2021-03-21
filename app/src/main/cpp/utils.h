#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <memory>
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

class Dims {
public:
    static const long PREFERRED_DIVISOR = 8;
    long w, h, d;
    bool operator==(const Dims & other) const;
    bool operator!=(const Dims & other) const;
    template<typename T> std::vector<T> toShape() const {
        return {T(1), T(h), T(w), T(d)};
    }
    cv::Size toCvSize() const;
    long numPixels() const;;
    Dims scaleToPixelBudget(long budget) const;
    Dims multiple(float x) const;
};

void floatsToLogits(const float* srcArray, const Dims &srcArrayDims, const cv::Size &intermediateSize, cv::Mat &intermediateSizeLogits);
void outputBytesToImg2C(const float* srcArray, cv::Mat &dstMat);
void matToFloats(cv::Mat &srcMat, cv::Mat& dstMat, const Dims requiredDims);
void softmax(cv::Mat &logits, float skyCoeff=1);

#endif /* UTILS_H */
