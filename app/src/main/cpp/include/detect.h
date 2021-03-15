/*
 * Copyright (C) 2012 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FILTERS_H
#define FILTERS_H

#include <jni.h>
#include <string.h>
#include <android/log.h>
#include <android/bitmap.h>

typedef unsigned int Color;

#define SetColor(a, r, g, b) ((a << 24) | (b << 16) | (g << 8) | (r << 0));
#define GetA(color) (((color) >> 24) & 0xFF)
#define GetB(color) (((color) >> 16) & 0xFF)
#define GetG(color) (((color) >> 8) & 0xFF)
#define GetR(color) (((color) >> 0) & 0xFF)

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

#define LOG(msg...) __android_log_print(ANDROID_LOG_VERBOSE, "NativeFilters", msg)

#define JNIFUNCF(cls, name, vars...) Java_com_example_face_ ## cls ## _ ## name(JNIEnv* env, jobject obj, vars)

#define RED i
#define GREEN i+1
#define BLUE i+2
#define ALPHA i+3
#define CLAMP(c, min, max) (MAX(min, MIN(c, max)))

__inline__ unsigned char clamp(int c);

__inline__ int clampMax(int c, int max);

/**
__inline__ unsigned char  clamp(int c){
    int N = 255;
    c &= ~(c >> 31);
    c -= N;
    c &= (c >> 31);
    c += N;
    return (unsigned char) c;
}

__inline__ int clampMax(int c,int max){
    c &= ~(c >> 31);
    c -= max;
    c &= (c >> 31);
    c += max;
    return  c;
}
*/
cv::Mat normalizeMat(cv::Mat in, float scale);
#endif // FILTERS_H
