package lutacc

import chisel3._
import chisel3.util._
import chisel3.util.HasBlackBoxResource
import freechips.rocketchip.config.{Config, Parameters}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile._

class LUTAccTimingIO extends Bundle {
  val clock = Input(Clock())
  val reset = Input(Bool())
  val start = Input(Bool())
  val designOmni = Input(Bool())
  val m = Input(UInt(32.W))
  val k = Input(UInt(32.W))
  val n = Input(UInt(32.W))
  val activationBits = Input(UInt(3.W))
  val outlierBasisPoints = Input(UInt(16.W))
  val busy = Output(Bool())
  val done = Output(Bool())
  val cycles = Output(UInt(64.W))
}

class LUTAccTimingBlackBox extends BlackBox with HasBlackBoxResource {
  val io = IO(new LUTAccTimingIO)
  addResource("/vsrc/LUTAccTimingBlackBox.sv")
}

class LUTAccRoCC(opcodes: OpcodeSet)(implicit p: Parameters)
    extends LazyRoCC(opcodes = opcodes, nPTWPorts = 0) {
  override lazy val module = new LUTAccRoCCModule(this)
}

class LUTAccRoCCModule(outer: LUTAccRoCC)(implicit p: Parameters)
    extends LazyRoCCModuleImp(outer) {
  val accelerator = Module(new LUTAccTimingBlackBox)
  val m = RegInit(1.U(32.W))
  val k = RegInit(4096.U(32.W))
  val n = RegInit(4096.U(32.W))
  val activationBits = RegInit(4.U(3.W))
  val outlierBasisPoints = RegInit(100.U(16.W))
  val responseValid = RegInit(false.B)
  val responseData = RegInit(0.U(64.W))
  val responseRd = RegInit(0.U(5.W))
  val waitingForAccelerator = RegInit(false.B)

  accelerator.io.clock := clock
  accelerator.io.reset := reset.asBool
  accelerator.io.start := false.B
  accelerator.io.designOmni := false.B
  accelerator.io.m := m
  accelerator.io.k := k
  accelerator.io.n := n
  accelerator.io.activationBits := activationBits
  accelerator.io.outlierBasisPoints := outlierBasisPoints

  io.cmd.ready := !waitingForAccelerator && !responseValid
  io.resp.valid := responseValid
  io.resp.bits.rd := responseRd
  io.resp.bits.data := responseData
  io.busy := waitingForAccelerator || responseValid || accelerator.io.busy
  io.interrupt := false.B

  io.mem.req.valid := false.B
  io.mem.req.bits := DontCare
  io.mem.s1_kill := false.B
  io.mem.s2_kill := false.B
  io.mem.keep_clock_enabled := false.B

  when (io.resp.fire()) {
    responseValid := false.B
  }

  when (io.cmd.fire()) {
    responseRd := io.cmd.bits.inst.rd
    switch (io.cmd.bits.inst.funct) {
      is (0.U) {
        m := io.cmd.bits.rs1(31, 0)
        k := io.cmd.bits.rs1(63, 32)
        n := io.cmd.bits.rs2(31, 0)
        activationBits := io.cmd.bits.rs2(33, 32) + 2.U
        outlierBasisPoints := io.cmd.bits.rs2(49, 34)
        responseData := 0.U
        responseValid := true.B
      }
      is (1.U) {
        accelerator.io.start := true.B
        accelerator.io.designOmni := false.B
        waitingForAccelerator := true.B
      }
      is (2.U) {
        accelerator.io.start := true.B
        accelerator.io.designOmni := true.B
        waitingForAccelerator := true.B
      }
      is (3.U) {
        responseData := io.busy.asUInt
        responseValid := true.B
      }
    }
  }

  when (accelerator.io.done) {
    waitingForAccelerator := false.B
    responseData := accelerator.io.cycles
    responseValid := true.B
  }
}

class WithLUTAcc extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      val lutacc = LazyModule(new LUTAccRoCC(OpcodeSet.custom3)(p))
      lutacc
    })
})
