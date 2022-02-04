lazy val deps = new {
  val keras       = "0.1.0-SNAPSHOT"
  val log         = "0.1.1"
  val tensorflow  = "0.4.0"
}

// change this according to your OS and CPU
val tfClassifier = "linux-x86_64"

lazy val root = project.in(file("."))
  .settings(
    name          := "tensorflow-java-test",
    organization  := "de.sciss",
    licenses      := Seq("LGPL v2.1+" -> url("http://www.gnu.org/licenses/lgpl-2.1.txt")),
    scalaVersion  := "3.1.1",
    libraryDependencies ++= Seq(
      "de.sciss"        %% "log"                  % deps.log,
      "de.sciss"        %% "tensorflow-keras"     % deps.keras,
      "org.tensorflow"  %  "tensorflow-core-api"  % deps.tensorflow,
      "org.tensorflow"  %  "tensorflow-core-api"  % deps.tensorflow classifier tfClassifier, // "linux-x86_64-mkl",
      "org.tensorflow"  %  "tensorflow-framework" % deps.tensorflow,
    )
  )
