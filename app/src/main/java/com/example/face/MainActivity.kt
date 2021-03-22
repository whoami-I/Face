package com.example.face

import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        var bitmap = BitmapFactory.decodeResource(resources,R.drawable.liuyifei3)
        Detector.getInstance().nativeInit(assets)
        Detector.getInstance().detectFace(bitmap)
    }
}