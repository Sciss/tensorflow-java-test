package de.sciss.tf

import org.tensorflow.keras
import org.tensorflow.keras.layers.{Conv, Conv2D, Dense, Layers}
import org.tensorflow.keras.models.{Model, Sequential}
import org.tensorflow.types.TFloat32

// implementation following this tutorial: https://www.tensorflow.org/tutorials/generative/dcgan
object CnnGanMnist:
  def main(args: Array[String]): Unit =
    val pair      = keras.datasets.MNIST.graphLoaders()
    val trainLoad = pair.first()
    val testLoad  = pair.second()

    ()

  end main

//  def makeGeneratorModel(): Model[TFloat32] =
//    val model = Sequential.of(classOf[TFloat32],
//      Layers.dense(7*7*256, /*Dense.Options.builder()*/ use_bias=false, input_shape=(100,???)),
//      Layers.batchNormalization(),
//      Layers.leakyReLU(),
//    )
//    model.add(Layers.reshape((7, 7, 256)))
//    assert (model.output_shape == (None, 7, 7, 256))  // Note: None is the batch size
//
//    model.add(Layers.conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=false))
//    assert (model.output_shape == (None, 7, 7, 128))
//    model.add(Layers.batchNormalization())
//    model.add(Layers.leakyReLU())
//
//    model.add(Layers.conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=false))
//    assert (model.output_shape == (None, 14, 14, 64))
//    model.add(Layers.batchNormalization())
//    model.add(Layers.leakyReLU())
//
//    model.add(Layers.conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=false, activation="tanh"))
//
//    assert (model.output_shape == (None, 28, 28, 1))
//
//    model
//  end makeGeneratorModel

  def makeDiscriminatorModel(): Model[TFloat32] =
    Sequential.of(classOf[TFloat32],
      Layers.input(28, 28, 1),
      Layers.conv2D(64, Seq(5, 5), Conv2D.Options(
        strides = Seq(2, 2),
        padding = Conv.Padding.Same,
//        inputShape = Array(28, 28, 1)
      )),
      ???, // Layers.leakyReLU(),
      ???, // Layers.dropout(0.3),

      Layers.conv2D(128, Seq(5, 5), Conv2D.Options(
        strides = Seq(2, 2),
        padding = Conv.Padding.Same,
      )),
      ???, // Layers.leakyReLU(),
      ???, // Layers.dropout(0.3),

      Layers.flatten(),
      Layers.dense(1),
    )
