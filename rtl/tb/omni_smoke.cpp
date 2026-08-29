#include "Vomni_rtl_smoke_top.h"
#include "verilated.h"

#include <cstdint>
#include <iostream>

static void tick(Vomni_rtl_smoke_top &dut) {
    dut.clk = 0;
    dut.eval();
    dut.clk = 1;
    dut.eval();
}

static std::int64_t sign_extend(std::uint64_t value, int bits) {
    const std::uint64_t mask = (std::uint64_t{1} << bits) - 1;
    value &= mask;
    const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
    return static_cast<std::int64_t>((value ^ sign) - sign);
}

static std::uint64_t encode_signed(std::int64_t value, int bits) {
    return static_cast<std::uint64_t>(value) & ((std::uint64_t{1} << bits) - 1);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Vomni_rtl_smoke_top dut;
    dut.reset = 1;
    dut.bqu_valid = 0;
    dut.lgu_valid = 0;
    dut.lut_write_enable = 0;
    dut.clear_accumulator = 0;
    dut.lookup_valid = 0;
    tick(dut);
    dut.reset = 0;

    dut.bqu_value = 28;
    dut.bit_width_minus_two = 2;  // four bits
    dut.scale_shift = 2;
    dut.bqu_valid = 1;
    dut.activation = 3;
    dut.centroid = 5;
    dut.correction = 2;
    dut.lgu_valid = 1;
    tick(dut);
    dut.bqu_valid = 0;
    dut.lgu_valid = 0;
    bool pass = dut.bqu_valid_out && dut.quantized_index == 15;
    pass &= sign_extend(dut.signed_code, 5) == 7;
    pass &= dut.lgu_valid_out && sign_extend(dut.lut_entry, 34) == 21;

    dut.lut_write_enable = 1;
    dut.lut_write_address = 1;
    dut.lut_write_data = encode_signed(10, 34);
    tick(dut);
    dut.lut_write_address = 2;
    dut.lut_write_data = encode_signed(-3, 34);
    tick(dut);
    dut.lut_write_enable = 0;
    dut.clear_accumulator = 1;
    tick(dut);
    dut.clear_accumulator = 0;
    dut.lookup_valid = 1;
    dut.lookup_index = 1;
    tick(dut);
    tick(dut);
    dut.lookup_index = 2;
    tick(dut);
    dut.lookup_valid = 0;
    pass &= sign_extend(dut.accumulator, 48) == 17;

    if (!pass) {
        std::cerr << "Omni-LUT RTL smoke failure: bqu_index="
                  << static_cast<int>(dut.quantized_index)
                  << " code=" << sign_extend(dut.signed_code, 5)
                  << " lgu=" << sign_extend(dut.lut_entry, 34)
                  << " accumulator=" << sign_extend(dut.accumulator, 48) << "\n";
        return 1;
    }
    std::cout << "Omni-LUT RTL smoke PASS\n";
    return 0;
}
