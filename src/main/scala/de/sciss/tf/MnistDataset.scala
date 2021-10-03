/*
 *  Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *  =======================================================================
 */

package de.sciss.tf

import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.ndarray.ByteNdArray
import org.tensorflow.ndarray.NdArrays

import java.io.{DataInputStream, FileNotFoundException, IOException}
import java.util.zip.GZIPInputStream
import org.tensorflow.ndarray.index.Indices.sliceFrom
import org.tensorflow.ndarray.index.Indices.sliceTo

/** Common loader and data preprocessor for MNIST and FashionMNIST datasets. */
object MnistDataset:
  val NUM_CLASSES = 10

  def create(validationSize       : Int,
             trainingImagesArchive: String,
             trainingLabelsArchive: String,
             testImagesArchive    : String,
             testLabelsArchive    : String): MnistDataset =
    val trainingImages  = readArchive(trainingImagesArchive)
    val trainingLabels  = readArchive(trainingLabelsArchive)
    val testImages      = readArchive(testImagesArchive)
    val testLabels      = readArchive(testLabelsArchive)
    if validationSize > 0 then
      new MnistDataset(
        trainingImages.slice(sliceFrom(validationSize)),
        trainingLabels.slice(sliceFrom(validationSize)),
        trainingImages.slice(sliceTo  (validationSize)),
        trainingLabels.slice(sliceTo  (validationSize)),
        testImages, testLabels)
    else
      new MnistDataset(trainingImages, trainingLabels, null, null, testImages, testLabels)

  private val TYPE_UBYTE = 0x08

  private def readArchive(archiveName: String) =
    val stream = getClass.getClassLoader.getResourceAsStream(archiveName)
    if (stream == null) throw new FileNotFoundException(archiveName)
    val archiveStream = new DataInputStream(new GZIPInputStream(stream))
    archiveStream.readShort() // first two bytes are always 0

    val magic = archiveStream.readByte()
    if (magic != TYPE_UBYTE) throw new IllegalArgumentException(s"\"$archiveName\" is not a valid archive")
    val numDims = archiveStream.readByte()
    val dimSizes = new Array[Long](numDims)
    var size = 1 // for simplicity, we assume that total size does not exceeds Integer.MAX_VALUE
    for (i <- 0 until dimSizes.length) {
      val sz = archiveStream.readInt()
      dimSizes(i) = sz
      size *= sz
    }
    val bytes = new Array[Byte](size)
    archiveStream.readFully(bytes)
    NdArrays.wrap(Shape.of(dimSizes: _*), DataBuffers.of(bytes, true, false))

class MnistDataset private(
                            trainingImages  : ByteNdArray,
                            trainingLabels  : ByteNdArray,
                            validationImages: ByteNdArray,
                            validationLabels: ByteNdArray,
                            testImages      : ByteNdArray,
                            testLabels      : ByteNdArray):
  private val _imageSize = trainingImages.get(0).shape.size

  def trainingBatches   (batchSize: Int): Iterator[ImageBatch] = new ImageBatchIterator(batchSize, trainingImages   , trainingLabels  )
  def validationBatches (batchSize: Int): Iterator[ImageBatch] = new ImageBatchIterator(batchSize, validationImages , validationLabels)
  def testBatches       (batchSize: Int): Iterator[ImageBatch] = new ImageBatchIterator(batchSize, testImages       , testLabels      )

  def testBatch: ImageBatch = new ImageBatch(testImages, testLabels)

  def imageSize: Long = _imageSize

  def numTrainingImages     : Int   = trainingImages.shape.size(0).toInt

  def numTrainingExamples   : Long  = trainingLabels   .shape.size(0)
  def numTestingExamples    : Long  = testLabels       .shape.size(0)
  def numValidationExamples : Long  = validationLabels .shape.size(0)
