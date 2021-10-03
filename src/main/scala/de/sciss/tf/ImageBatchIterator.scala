package de.sciss.tf

import org.tensorflow.ndarray.ByteNdArray
import org.tensorflow.ndarray.index.Indices.range
import org.tensorflow.ndarray.index.{Index, Indices}

import java.util

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


/** Basic batch iterator across images presented in datset. */
class ImageBatchIterator(batchSize: Int, images: ByteNdArray, labels: ByteNdArray)
  extends Iterator[ImageBatch] {

  private var numImages   = if images != null then images.shape.size(0) else 0L
  private var batchStart  = 0

  override def hasNext: Boolean = batchStart < numImages

  override def next(): ImageBatch = {
    val nextBatchSize = math.min(batchSize, numImages - batchStart).toInt
    val range         = Indices.range(batchStart, batchStart + nextBatchSize)
    batchStart       += nextBatchSize
    new ImageBatch(images.slice(range), labels.slice(range))
  }
}
