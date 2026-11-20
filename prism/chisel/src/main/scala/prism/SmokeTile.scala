package prism

import chisel3._

/** Minimal parameterized MAC tile used to validate the Chisel toolchain. */
class SmokeTile(rows: Int = 2, cols: Int = 2, bits: Int = 8, accBits: Int = 24) extends Module {
  require(rows > 0 && cols > 0 && bits > 0 && accBits > 0)

  val io = IO(new Bundle {
    val clear = Input(Bool())
    val valid = Input(Bool())
    val activation = Input(Vec(rows, SInt(bits.W)))
    val weight = Input(Vec(rows, Vec(cols, SInt(bits.W))) )
    val psum = Output(Vec(rows, Vec(cols, SInt(accBits.W))))
  })

  val accum = RegInit(VecInit(Seq.fill(rows)(VecInit(Seq.fill(cols)(0.S(accBits.W))))))
  when (io.clear) {
    for (row <- 0 until rows; col <- 0 until cols) {
      accum(row)(col) := 0.S
    }
  }.elsewhen (io.valid) {
    for (row <- 0 until rows; col <- 0 until cols) {
      accum(row)(col) := accum(row)(col) + io.activation(row) * io.weight(row)(col)
    }
  }
  io.psum := accum
}

object SmokeTileMain extends App {
  emitVerilog(new SmokeTile(), Array("--target-dir", "generated"))
}
