lazy val deps = new {
  val log         = "0.1.1"
  val tensorflow  = "0.3.3"
}

// change this according to your OS and CPU
val tfClassifer = "linux-x86_64"

lazy val root = project.in(file("."))
  .settings(
    name          := "tensorflow-java-test",
    organization  := "de.sciss",
    licenses      := Seq("LGPL v2.1+" -> url("http://www.gnu.org/licenses/lgpl-2.1.txt")),
    scalaVersion  := "3.0.2",
    libraryDependencies ++= Seq(
      "de.sciss"        %% "log"                  % deps.log,
      "org.tensorflow"  %  "tensorflow-core-api"  % deps.tensorflow,
      "org.tensorflow"  %  "tensorflow-core-api"  % deps.tensorflow classifier tfClassifer, // "linux-x86_64-mkl",
      "org.tensorflow"  %  "tensorflow-framework" % deps.tensorflow,
    )
  )
