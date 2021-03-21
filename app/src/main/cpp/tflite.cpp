#include "tflite.h"
#include "logutils.h"

void TfBackend::loadPmodel(const char *file_buffer, size_t size){
    pmodel = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(file_buffer,size);
    if (!pmodel) {
        throw std::runtime_error("Failed to mmap model");
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder{*pmodel, resolver}(&pmodel_interpreter);
    if (!pmodel_interpreter) {
        throw std::runtime_error("Failed to construct TFLite interpreter");
    }

    pmodel_interpreter->SetNumThreads(num_threads);
    //VerifyShapes();
    if (pmodel_interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Could not allocate tensors");
    }
}

void TfBackend::loadRmodel(const char *file_buffer, size_t size){
    rmodel = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(file_buffer,size);
    if (!rmodel) {
        throw std::runtime_error("Failed to mmap model");
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder{*rmodel, resolver}(&rmodel_interpreter);
    if (!rmodel_interpreter) {
        throw std::runtime_error("Failed to construct TFLite interpreter");
    }

    rmodel_interpreter->SetNumThreads(num_threads);
    //VerifyShapes();
    if (rmodel_interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Could not allocate tensors");
    }
}

void TfBackend::loadOmodel(const char *file_buffer, size_t size){
    omodel = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(file_buffer,size);
    if (!omodel) {
        throw std::runtime_error("Failed to mmap model");
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder{*omodel, resolver}(&omodel_interpreter);
    if (!omodel_interpreter) {
        throw std::runtime_error("Failed to construct TFLite interpreter");
    }

    omodel_interpreter->SetNumThreads(num_threads);
    //VerifyShapes();
    if (omodel_interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Could not allocate tensors");
    }
}

float* TfBackend::getPmodelInputPtr() {
    return pmodel_interpreter->typed_input_tensor<float>(0);
}

float* TfBackend::getRmodelInputPtr() {
    return rmodel_interpreter->typed_input_tensor<float>(0);
}

float* TfBackend::getOmodelInputPtr() {
    return omodel_interpreter->typed_input_tensor<float>(0);
}

TfBackend::TfBackend() {

}


