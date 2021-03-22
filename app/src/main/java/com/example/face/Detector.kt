package com.example.face

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class Detector {

    companion object {
        @Volatile
        private var instance: Detector? = null
        private lateinit var context: Context
        val P_MODEL_FILE: String = "pmodel.tflite"
        val R_MODEL_FILE: String = "rmodel.tflite"
        val O_MODEL_FILE: String = "omodel.tflite"
        val IMAGE_WIDTH = 28
        val IMAGE_HEIGHT = 28
        fun getInstance(): Detector {
            return instance ?: synchronized(this) {
                instance ?: Detector().also { instance = it }
            }
        }

        fun setContext(context: Context) {
            this.context = context
        }
    }

    lateinit var pModel: Interpreter
    lateinit var rModel: Interpreter
    lateinit var oModel: Interpreter

    private var result = Array(1) { FloatArray(10) }

    private var imgData: ByteBuffer
    private val intValues = IntArray(IMAGE_HEIGHT * IMAGE_HEIGHT)

    constructor() {

        imgData = ByteBuffer.allocateDirect(IMAGE_HEIGHT * IMAGE_WIDTH * 4)
            .order(ByteOrder.nativeOrder())
    }

    init {
        System.loadLibrary("filters")
    }

    fun loadModel() {
        val gpuDelegate = GpuDelegate()
        val tfliteOptions =
            Interpreter.Options()
        tfliteOptions.addDelegate(gpuDelegate)
        tfliteOptions.setNumThreads(2)
        var tfliteModel = loadModelFile(context, P_MODEL_FILE)
        pModel = Interpreter(tfliteModel, tfliteOptions)
        tfliteModel = loadModelFile(context, R_MODEL_FILE)
        rModel = Interpreter(tfliteModel, tfliteOptions)
        tfliteModel = loadModelFile(context, O_MODEL_FILE)
        oModel = Interpreter(tfliteModel, tfliteOptions)
    }


    /**
     * 传进来必须是一张灰度图
     */
    fun getNumber(bitmap: Bitmap): Int {
        if (bitmap.width != IMAGE_WIDTH || bitmap.height != IMAGE_HEIGHT) return -1
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        imgData.rewind()
        for (i in 0 until bitmap.height) {
            for (j in 0 until bitmap.width) {
                val color = intValues[pixel++]
                val rColor = (color and 0xFF)
                imgData.putFloat(rColor.toFloat())
            }
        }

        pModel.run(imgData, result)
        result.forEach {
            it.forEach { Log.d("TAGTAG", " result is ${it}") }
        }
        var largest = Float.MIN_VALUE
        var index = -1
        for (i in 0 until result[0].size) {
            if (result[0][i] > largest) {
                largest = result[0][i]
                index = i
            }
        }
        return index
    }

    fun detectFace(bitmap: Bitmap) {
        nativeDetectFace(bitmap, bitmap.width, bitmap.height)
    }
    external fun nativeInit(assetManager: AssetManager)
    external fun nativeDetectFace(bitmap: Bitmap, width: Int, height: Int)

    /** Memory-map the model file in Assets.  */
    @Throws(IOException::class)
    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream =
            FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )
    }
}