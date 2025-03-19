package com.detection.detectionobject

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaPlayer
import android.os.Bundle
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.detection.detectionobject.Constants.LABELS_PATH
import com.detection.detectionobject.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.R
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding // View binding that accesses UI components

    private val isFrontCamera = false // value that determines if the front camera is used

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null // camera X components

    private var detectionCounter = 0  // verify detection with timers
    private var lastDetectionTime = System.currentTimeMillis()
    private val detectionThreshold = 8 // number of detections required before action is taken
    private val timeWindow = 2000L // counts detections
    private var mediaPlayer: MediaPlayer? = null // used to play alert sounds

    private lateinit var detector: Detector // object  detection model instance
    private lateinit var cameraExecutor: ExecutorService // background thread executor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this) // initializes the detector
        detector.setup()

        if (allPermissionsGranted()) { //checks if permissions are already granted, if not asks for permissions
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor() // starts a background thread
    }

    private fun startCamera() { // initializes the camera
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider  = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {  // binds camera preview
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK) // selects the back camera
            .build()

        preview =  Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy -> // analyses each frame from the camera
            val bitmapBuffer =
                Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply { // rotates and mirrors the image
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

                if (isFrontCamera) {
                    postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                    )
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            detector.detect(rotatedBitmap) // runs object detection
        }

        cameraProvider.unbindAll() // binds the camera provider to lifecycle

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch(exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { //checks if camera permissions are granted
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult( // handles permission requests
        ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.clear() // releases model resources
        cameraExecutor.shutdown() // stops the background thread
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() { // clears ui when no objects are detected
        runOnUiThread {
            binding.overlay.setResults(emptyList())
        }
    }

    //processes detected objects and plays an audio alert
    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }

            // makes sure the detection is real based on frequency
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastDetectionTime <= timeWindow) {
                detectionCounter++
            } else {
                detectionCounter = 1
            }
            lastDetectionTime = currentTime

            if (detectionCounter >= detectionThreshold) {
                for (boundingBox in boundingBoxes) {
                    val detectedClassName = boundingBox.clsName

                    // plays an audio file based on the detected class
                    val audioResource = when (detectedClassName) {
                        "texting" -> R.raw.texting
                        "talking on a phone" -> R.raw.talkingonaphone
                        "drinking" -> R.raw.drinking
                        "eating" -> R.raw.eating
                        "smoking" -> R.raw.smoking
                        else -> Log.d("Detection", "Nothing detected")
                    }

                    // prevents 2 or more audios playing at the same time
                    if (mediaPlayer == null || !mediaPlayer!!.isPlaying) {
                        mediaPlayer = MediaPlayer.create(applicationContext, audioResource)
                        mediaPlayer?.start()

                        // releases the media player
                        mediaPlayer?.setOnCompletionListener {
                            it.release()
                            mediaPlayer = null
                        }
                    }
                }
            }
        }
    }
}
