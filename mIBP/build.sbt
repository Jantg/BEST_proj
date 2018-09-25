name := "mIBP"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0",
  "org.apache.spark" %% "spark-sql" % "2.2.0"
  //"org.scalaz" %% "scalaz-core" % "7.3.0-M25"
  //"graphframes" % "graphframes" % "0.5.0-spark2.1-s_2.11"
)