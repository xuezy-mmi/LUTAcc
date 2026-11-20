ThisBuild / scalaVersion := "2.13.14"
ThisBuild / organization := "prism"
ThisBuild / version := "0.1.0"

lazy val root = (project in file("."))
  .settings(
    name := "prism-chisel",
    libraryDependencies += "org.chipsalliance" %% "chisel" % "6.6.0",
    libraryDependencies += compilerPlugin("org.chipsalliance" % "chisel-plugin" % "6.6.0" cross CrossVersion.full),
    Compile / run / fork := true,
    scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked")
  )
