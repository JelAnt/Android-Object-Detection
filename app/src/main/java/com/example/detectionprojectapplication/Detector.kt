package com.example.detectionprojectapplication

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader // importing libraries

class Detector( // initializing variables
    private val context: Context, // holds assets
    private val modelPath: String, // holds path to yolo model
    private val labelPath: String, // holds path to yolo model's labels
    private val detectorListener: DetectorListener // keeps track of detections
) {

    private var interpreter: Interpreter? = null // tensorflow interpreter for inference
    private var labels = mutableListOf<String>() // holds class labels

    private var tensorWidth = 0
    private var tensorHeight = 0 // image input size
    private var numChannel = 0 // number of output classes
    private var numElements = 0 // number of output detection boxes

    private val imageProcessor = ImageProcessor.Builder() // image processing pipeline
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION)) // normalizes pixel values
        .add(CastOp(INPUT_IMAGE_TYPE)) // converts image data to float32
        .build() // finalizes build

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath) // loads tflite model from the assets folder
        val options = Interpreter.Options() // object that controls tflite interpreter options
        options.numThreads = 4  // sets inference to run on 4 cpu threads
        interpreter = Interpreter(model, options) // initializes tflite interpreter

        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return // gets shape of the input image
        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return // gets shape of the detection results

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1]
        numElements = outputShape[2] // stores input and output dimensions

        try {
            val inputStream: InputStream = context.assets.open(labelPath) // opens labels file
            val reader = BufferedReader(InputStreamReader(inputStream)) // prepares input reader

            var line: String? = reader.readLine()
            while (line != null && line != "") { // assigns lines from the label txt to labels
                labels.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) { // prints stack if the label txt fails to load
            e.printStackTrace()
        }
    }

    fun clear() {
        interpreter?.close() // frees memory used by the model
        interpreter = null // marks interpreter as null
    }

    fun detect(frame: Bitmap) {
        interpreter ?: return // exits if the model isn't loaded
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return // exits if any tensor shape value is 0

        var inferenceTime = SystemClock.uptimeMillis() // times inference

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false) // resizes the input image

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap) // converts resized input image to a tensor image
        val processedImage = imageProcessor.process(tensorImage) // runs preprocessing on the tensor image
        val imageBuffer = processedImage.buffer // holds the image buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1 , numChannel, numElements), OUTPUT_IMAGE_TYPE) // stores detection results
        interpreter?.run(imageBuffer, output.buffer) // runs inference


        val bestBoxes = bestBox(output.floatArray) // stores best box detections
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime // calculates how long inference took


        if (bestBoxes == null) { // if no detection
            detectorListener.onEmptyDetect() // run function for no detections
            return
        }

        detectorListener.onDetect(bestBoxes, inferenceTime) // passes detection results
    }

    private fun bestBox(array: FloatArray) : List<BoundingBox>? { // takes in model output and returns a list of bounding boxes

        val boundingBoxes = mutableListOf<BoundingBox>() // list to store bounding boxes

        for (c in 0 until numElements) { // loops through each detection and assigns conf scores
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel){
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) { // keeps conf scores above 0.3
                val clsName = labels[maxIdx] // retrieves class label from labels
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3] // holds coordinates and size
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F) // calculates bounding box coordinates
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue // filters invalid bounding box coordinates

                boundingBoxes.add( // adds bounding box coords to the list
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null // returns null if there are no valid bounding boxes

        return applyNMS(boundingBoxes) // calls applyNMS function to remove duplicate boxes
    }

    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> { // takes a list of bounding boxes, removes duplicates and returns a new list
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList() //sorts bounding boxes by highest confidence
        val selectedBoxes = mutableListOf<BoundingBox>()  // list for filtered bounding boxes

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first) // adds highest confidence boxes to the final list and removes them from sorted boxes list

            val iterator = sortedBoxes.iterator() // loops over the remaining boxes and removes boxes with an IOU above 0.5 to reduce duplicates
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes // returns filtered boxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float { // calculates IOU between two bounding boxes
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2) // finds coords between box 1 and box 2
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1) // calculates intersection area
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea) // calculates and returns IOU
    }

    interface DetectorListener { // defines a listener inference
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object { // holding constant values
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
    }
}