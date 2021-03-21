#ifndef TFLITE_BACKEND_H
#define TFLITE_BACKEND_H

#include <memory>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

class TfBackend {
// /nas/projects/on-device-ai/benchmarks/models/THEBEST99k_MACE/tflite/models/THEBEST99k_MACE-float32.lite

    std::string imageNode = "ImageTensor";
    std::string outputNode = "SemanticPredictions";

    size_t num_threads = 2;

public:
//    void invoke();

    TfBackend();

    void loadPmodel(const char *file_buffer, size_t size);

    void loadOmodel(const char *file_buffer, size_t size);

    void loadRmodel(const char *file_buffer, size_t size);

    float *getPmodelInputPtr();

    float *getRmodelInputPtr();

    float *getOmodelInputPtr();

    std::unique_ptr<tflite::FlatBufferModel> pmodel;
    std::unique_ptr<tflite::Interpreter> pmodel_interpreter;

    std::unique_ptr<tflite::FlatBufferModel> rmodel;
    std::unique_ptr<tflite::Interpreter> rmodel_interpreter;

    std::unique_ptr<tflite::FlatBufferModel> omodel;
    std::unique_ptr<tflite::Interpreter> omodel_interpreter;
};

#endif
