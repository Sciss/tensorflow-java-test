# tensorflow-java-test

[![Build Status](https://github.com/Sciss/tensorflow-java-test/workflows/Scala%20CI/badge.svg?branch=main)](https://github.com/Sciss/tensorflow-java-test/actions?query=workflow%3A%22Scala+CI%22)

## statement

This is a project for testing the [Java bindings](https://github.com/tensorflow/java) to TensorFlow, using Scala 3.
It is (C)opyright 2021–2022 by Hanns Holger Rutz. All rights reserved. The project is released under 
the [GNU Lesser General Public License](https://raw.github.com/Sciss/tensorflow-java-test/main/LICENSE) v2.1+ and
comes with absolutely no warranties. To contact the author, send an e-mail to `contact at sciss.de`.

## requirements / installation

The project build with [sbt](https://www.scala-sbt.org/). You may need to edit `build.sbt` to adjust `tfClassifer`.

Furthermore, for Keras models, it currently relies on an unstable version
of [tensorflow-keras-scala](https://codeberg.org/sciss/tensorflow-keras-scala) (bindings). To prepare them, clone
that repository, and within use `sbt +publishLocal` to make an unstable version available to this build.

Then here,

`sbt run`

`sbt 'runMain de.sciss.tf.CnnTest 10 100 sgd'`
