package de.sciss.tf

import org.tensorflow.keras.activations.Activations
import org.tensorflow.{Graph, Operand, keras}
import org.tensorflow.keras.layers.{BatchNormalization, Conv, Conv2D, Dense, Layer, Layers}
import org.tensorflow.keras.losses.Losses
import org.tensorflow.keras.metrics.Metrics
import org.tensorflow.keras.models.{Model, Sequential}
import org.tensorflow.keras.optimizers.Optimizers
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.{TFloat32, TInt32}

import scala.jdk.CollectionConverters.*

// implementation following this tutorial: https://www.tensorflow.org/tutorials/generative/dcgan
object CnnGanMnist:
  def main(args: Array[String]): Unit =
    val (trainLoad, testLoad) = keras.datasets.MNIST.graphLoaders

    println("makeGeneratorModel")
    val generator = makeGeneratorModel()
    val graph     = new Graph
    val tf        = Ops.create(graph)
    val noise     = tf.random.randomStandardNormal(tf.constant(Array(1, 100)), classOf[TFloat32])

    println("generator.compile")
    generator.compile(tf, Optimizers.select(Optimizers.sgd), Losses.select(Losses.sparseCategoricalCrossentropy),
      Seq(Metrics.select(Metrics.accuracy)).asJava)
    // generator

    println("generator.call")
    val generatedImage: Operand[TFloat32] = generator(tf, noise) // , training = false)
    // plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    println(generatedImage)
    println(generatedImage.shape())
    // assert (generatedImage.shape == (None, 28, 28, 1))

    ()

  end main

  final val BUFFER_SIZE = 60000
  final val BATCH_SIZE  = 256

  def makeGeneratorModel(): Model[TFloat32] =
    val model = Sequential(classOf[TFloat32],
//      Layers.input(Shape.of(100)),
      Layers.input(Shape.of(100), batchSize = 10),    // XXX TODO: tutorial doesn't specify batch size
      Layers.dense(7*7*256, useBias = false),
      Layers.batchNormalization(),
      Layers.leakyReLU(),

      Layers.reshape(Shape.of(7, 7, 256)),
      // assert (model.output_shape == (None, 7, 7, 256))  // Note: None is the batch size

      Layers.conv2DTranspose(128, (5, 5), strides = (1, 1), padding = Conv.Padding.Same, useBias = false),
      // assert (model.output_shape == (None, 7, 7, 128))
      Layers.batchNormalization(),
      Layers.leakyReLU(),

      Layers.conv2DTranspose(64, (5, 5), strides = (2, 2), padding = Conv.Padding.Same, useBias = false),
      // assert (model.output_shape == (None, 14, 14, 64))
      Layers.batchNormalization(),
      Layers.leakyReLU(),

      Layers.conv2DTranspose(1, (5, 5), strides = (2, 2), padding = Conv.Padding.Same, useBias = false,
        activation = Some(Activations.tanh)
      ),
    )
    // assert (model.output_shape == (None, 28, 28, 1))
    model
  end makeGeneratorModel

  def makeDiscriminatorModel(): Model[TFloat32] =
    Sequential(classOf[TFloat32],
      Layers.input(Shape.of(28, 28, 1)),
      Layers.conv2D( 64, (5, 5), strides = (2, 2), padding = Conv.Padding.Same),
      Layers.leakyReLU(),
      Layers.dropout(0.3),

      Layers.conv2D(128, (5, 5), strides = (2, 2), padding = Conv.Padding.Same),
      Layers.leakyReLU(),
      Layers.dropout(0.3),

      Layers.flatten(),
      Layers.dense(1),
    )
