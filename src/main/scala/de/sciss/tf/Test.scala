package de.sciss.tf

import org.tensorflow.op.Ops
import org.tensorflow.proto.framework.DataType
import org.tensorflow.types.TFloat64
import org.tensorflow.{EagerSession, Graph, Session, Tensor}

object Test:
  def main(args: Array[String]): Unit =
    val g       = new Graph()
    val ops     = Ops.create(g)
    import ops.*
    val a       = constant(3.0)
    val b       = constant(2.0)
    val x       = placeholder(classOf[TFloat64])
    val y       = placeholder(classOf[TFloat64])
    val ax      = math.mul(a, x)
    val by      = math.mul(b, y)
    val z       = math.add(ax, by)
    val session = new Session(g)
    val r       = session.runner()
    val res     = r.fetch(z)
      .feed(x, TFloat64.scalarOf(3.0))
      .feed(y, TFloat64.scalarOf(6.0))
      .run
      .get(0)

    res match
      case tf: TFloat64 => println(tf.getDouble())  // 21.0
      case other        => println(s"OTHER: $other")
