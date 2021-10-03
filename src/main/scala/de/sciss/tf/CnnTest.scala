package de.sciss.tf

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

import de.sciss.log.{Level, Logger}
import org.tensorflow.{Graph, Session}
import org.tensorflow.framework.optimizers.{AdaDelta, AdaGrad, AdaGradDA, Adam, GradientDescent, Momentum, RMSProp}
import org.tensorflow.ndarray.index.Indices
import org.tensorflow.ndarray.{FloatNdArray, Shape}
import org.tensorflow.op.Ops
import org.tensorflow.op.core.{Placeholder, Reshape}
import org.tensorflow.op.random.TruncatedNormal
import org.tensorflow.types.{TFloat32, TUint8}

import java.util
import scala.collection.JavaConverters.*

/** Builds a LeNet-5 style CNN for MNIST.
  *
  * Scala translation of https://github.com/tensorflow/java-models/blob/master/tensorflow-examples/src/main/java/org/tensorflow/model/examples/cnn/lenet/CnnMnist.java
  * originally published under Apache 2.0 license
  */
object CnnTest:
  val INPUT_NAME    = "input"
  val OUTPUT_NAME   = "output"
  val TARGET        = "target"
  val TRAIN         = "train"
  val TRAINING_LOSS = "training_loss"
  val INIT          = "init"

  val IMAGE_SIZE    = 28
  val NUM_CHANNELS  = 1
  val PIXEL_DEPTH   = 255

  val SEED          = 123456789L
  val PADDING_TYPE  = "SAME"

  val NUM_LABELS    = MnistDataset.NUM_CLASSES

  private val logger = new Logger("cnn", Level.Info, Console.err)

  private val TRAINING_IMAGES_ARCHIVE = "mnist/train-images-idx3-ubyte.gz"
  private val TRAINING_LABELS_ARCHIVE = "mnist/train-labels-idx1-ubyte.gz"
  private val TEST_IMAGES_ARCHIVE     = "mnist/t10k-images-idx3-ubyte.gz"
  private val TEST_LABELS_ARCHIVE     = "mnist/t10k-labels-idx1-ubyte.gz"

  def main(args: Array[String]): Unit =
    logger.info("Usage: CnnTest <num-epochs> <minibatch-size> <optimizer-name = adadelta | adagradda | adagrad | adam | sgd | momentum | rmsprop>")

    val dataset = MnistDataset.create(0,
      TRAINING_IMAGES_ARCHIVE,
      TRAINING_LABELS_ARCHIVE,
      TEST_IMAGES_ARCHIVE,
      TEST_LABELS_ARCHIVE
    )

    logger.info(s"Loaded data. numTrainingImages ${dataset.numTrainingImages}")
    val epochs        = args(0).toInt
    val minibatchSize = args(1).toInt

    val graph = build(args(2))
    try
      val session = new Session(graph)
      try
        train(session, epochs, minibatchSize, dataset)
        logger.info("Trained model")
        test(session, minibatchSize, dataset)
      finally
        session.close()
    finally
      graph.close()
  end main

  def build(optimizerName: String): Graph =
    val g       = new Graph()
    val ops     = Ops.create(g)
    import ops.*

    // ---- inputs ----
    val input0        = withName(INPUT_NAME).placeholder(classOf[TUint8],
      Placeholder.shape(Shape.of(-1, IMAGE_SIZE, IMAGE_SIZE))
    )
    val input         = reshape(input0, array(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    val labels        = withName(TARGET).placeholder(classOf[TUint8])

    // ---- scaling the features ----
    val scaleAdd      = constant(-PIXEL_DEPTH / 2.0f)
    val scaleMul      = constant(1.0f / PIXEL_DEPTH)
    val scaled        = math.mul(math.add(dtypes.cast(input, classOf[TFloat32]), scaleAdd), scaleMul)

    // ---- first convolution layer ----
    val conv1Weights  = variable(math.mul(
      random.truncatedNormal(array(5, 5, NUM_CHANNELS, 32), classOf[TFloat32], TruncatedNormal.seed(SEED)),
      constant(0.1f)
    ))
    val conv1         = nn.conv2d(scaled, conv1Weights, util.Arrays.asList(1L, 1L, 1L, 1L), PADDING_TYPE)
    val conv1Biases   = variable(fill(array(32), constant(0.0f)))
    val relu1         = nn.relu(nn.biasAdd(conv1, conv1Biases))

    // ---- first pooling layer ----
    val pool1 = nn.maxPool(relu1, array(1, 2, 2, 1), array(1, 2, 2, 1), PADDING_TYPE)

    // ---- second convolution layer ----
    val conv2Weights  = variable(math.mul(
      random.truncatedNormal(array(5, 5, 32, 64), classOf[TFloat32], TruncatedNormal.seed(SEED)),
      constant(0.1f)
    ))
    val conv2         = nn.conv2d(pool1, conv2Weights, util.Arrays.asList(1L, 1L, 1L, 1L), PADDING_TYPE)
    val conv2Biases   = variable(fill(array(64), constant(0.1f)))
    val relu2         = nn.relu(nn.biasAdd(conv2, conv2Biases))

    // ---- second pooling layer ----
    val pool2         = nn.maxPool(relu2, array(1, 2, 2, 1), array(1, 2, 2, 1), PADDING_TYPE)
    
    // ---- flatten inputs ----
    val flatten       = reshape(pool2,
      concat(
        util.Arrays.asList(slice(shape(pool2), array(0), array(1)), array(-1)),
        constant(0))
    )
    
    // ---- fully connected layer ----
    val fc1Weights    = variable(math.mul(
      random.truncatedNormal(array(IMAGE_SIZE * IMAGE_SIZE * 4, 512), classOf[TFloat32], TruncatedNormal.seed(SEED)),
      constant(0.1f)
    ))
    val fc1Biases     = variable(fill(array(512), constant(0.1f)))
    val relu3         = nn.relu(math.add(linalg.matMul(flatten, fc1Weights), fc1Biases))
    
    // ---- softmax layer ----
    val fc2Weights    = variable(math.mul(random.truncatedNormal(array(512, NUM_LABELS), classOf[TFloat32], TruncatedNormal.seed(SEED)), constant(0.1f)))
    val fc2Biases     = variable(fill(array(NUM_LABELS), constant(0.1f)))
    
    val logits        = math.add(linalg.matMul(relu3, fc2Weights), fc2Biases)

    // ---- predicted outputs ----
    val prediction    = withName(OUTPUT_NAME).nn.softmax(logits)

    // ---- loss function and regularization ----
    val hot           = oneHot(labels, constant(10), constant(1.0f), constant(0.0f))
    val batchLoss     = nn.raw.softmaxCrossEntropyWithLogits(logits, hot)
    val labelLoss     = math.mean(batchLoss.loss, constant(0))
    val regularizers  = math.add(
      nn.l2Loss(fc1Weights),
      math.add(nn.l2Loss(fc1Biases), math.add(nn.l2Loss(fc2Weights), nn.l2Loss(fc2Biases)))
    )
    val loss          = withName(TRAINING_LOSS).math.add(labelLoss, math.mul(regularizers, constant(5e-4f)))

    val lcOptimizerName = optimizerName.toLowerCase
    // Optimizer
    val optimizer = lcOptimizerName match
      case "adadelta" =>
        new AdaDelta(g, 1f, 0.95f, 1e-8f)
      case "adagradda" =>
        new AdaGradDA(g, 0.01f)
      case "adagrad" =>
        new AdaGrad(g, 0.01f)
      case "adam" =>
        new Adam(g, 0.001f, 0.9f, 0.999f, 1e-8f)
      case "sgd" =>
        new GradientDescent(g, 0.01f)
      case "momentum" =>
        new Momentum(g, 0.01f, 0.9f, false)
      case "rmsprop" =>
        new RMSProp(g, 0.01f, 0.9f, 0.0f, 1e-10f, false)
      case _ =>
        throw new IllegalArgumentException(s"Unknown optimizer $optimizerName")

    logger.info(s"Optimizer = $optimizer")
    /*val minimize =*/ optimizer.minimize(loss, TRAIN)

    init()
    g

  end build

  def train(session: Session, epochs: Int, minibatchSize: Int, dataset: MnistDataset): Unit =
    // Initialises the parameters.
    session.runner.addTarget(INIT).run
    logger.info("Initialised the model parameters")
    var interval = 0
    // Train the model
    for epoch <- 0 until epochs do
      logger.info(s"epoch = $epoch / $epochs")
      dataset.trainingBatches(minibatchSize).foreach { trainingBatch =>
        val batchImages = TUint8.tensorOf(trainingBatch.images)
        try
          val batchLabels = TUint8.tensorOf(trainingBatch.labels)
          try
            val loss = session.runner
              .feed(TARGET, batchLabels)
              .feed(INPUT_NAME, batchImages)
              .addTarget(TRAIN)
              .fetch(TRAINING_LOSS)
              .run
              .get(0).asInstanceOf[TFloat32]
            if (interval % 100 == 0) logger.info(s"Iteration = $interval, training loss = ${loss.getFloat()}")
            loss.close()
          finally
            batchLabels.close()
        finally
          batchImages.close()

        interval += 1
      }

  end train

  def test(session: Session, minibatchSize: Int, dataset: MnistDataset): Unit =
    var correctCount    = 0
    val confusionMatrix = Array.ofDim[Int](10, 10)
    dataset.testBatches(minibatchSize).foreach { trainingBatch =>
      val transformedInput = TUint8.tensorOf(trainingBatch.images)
      try
        val out = session.runner
          .feed(INPUT_NAME, transformedInput)
          .fetch(OUTPUT_NAME)
          .run
          .get(0).asInstanceOf[TFloat32]
        try
          val labelBatch = trainingBatch.labels
          var k = 0
          while k < labelBatch.shape.size(0) do
            val trueLabel = labelBatch.getByte(k)
            var predLabel = 0
            predLabel = argmax(out.slice(Indices.at(k), Indices.all))
            if (predLabel == trueLabel) correctCount += 1
            confusionMatrix(trueLabel)(predLabel) += 1
            k += 1

        finally
          out.close()
      finally
        transformedInput.close()
    }
    logger.info(f"Final accuracy = ${correctCount.toDouble * 100 / dataset.numTestingExamples}%1.1f%%")
    val sb = new StringBuilder
    sb.append("Label")
    for (i <- 0 until confusionMatrix.length) {
      sb.append(String.format("%1$5s", "" + i))
    }
    sb.append("\n")
    for (i <- 0 until confusionMatrix.length) {
      sb.append(String.format("%1$5s", "" + i))
      for (j <- 0 until confusionMatrix(i).length) {
        sb.append(String.format("%1$5s", "" + confusionMatrix(i)(j)))
      }
      sb.append("\n")
    }
    println(sb.toString)

  end test

  /** Finds the maximum probability and return it's index.
    *
    * @param probabilities The probabilites.
    * @return The index of the max.
    */
  def argmax(probabilities: FloatNdArray): Int =
    var maxVal = Float.NegativeInfinity
    var idx = 0
    var i = 0
    while (i < probabilities.shape.size(0)) {
      val curVal = probabilities.getFloat(i)
      if (curVal > maxVal) {
        maxVal = curVal
        idx = i
      }

      i += 1
    }
    idx
