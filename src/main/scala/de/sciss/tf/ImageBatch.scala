package de.sciss.tf

import org.tensorflow.ndarray.ByteNdArray

class ImageBatch(val images: ByteNdArray, val labels: ByteNdArray)
