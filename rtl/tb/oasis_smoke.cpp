#include "Voasis_rtl_smoke_top.h"
#include "verilated.h"

#include <cstdint>
#include <iostream>

static void tick(Voasis_rtl_smoke_top &dut) {
    dut.clk = 0;
    dut.eval();
    dut.clk = 1;
    dut.eval();
}

static std::uint64_t pack16(const std::int16_t *values, int count) {
    std::uint64_t packed = 0;
    for (int index = 0; index < count; ++index) {
        packed |= static_cast<std::uint64_t>(static_cast<std::uint16_t>(values[index]))
                  << (16 * index);
    }
    return packed;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Voasis_rtl_smoke_top dut;
    dut.reset = 1;
    dut.valid_in = 0;
    tick(dut);
    dut.reset = 0;

    // Four 2-bit lanes: concat {activation, weight} => 3, 6, 9, 12.
    dut.activation_indices = 0b11100100;
    dut.weight_indices = 0b00011011;
    const std::uint16_t expected_concat = 0xC963;

    const std::int16_t boundaries[3] = {-10, 0, 10};
    dut.boundaries = pack16(boundaries, 3);
    dut.sample = 0;

    const std::int16_t values[4] = {5, -7, 12, 12};
    dut.values = pack16(values, 4);
    dut.valid_in = 1;
    tick(dut);
    dut.valid_in = 0;

    bool pass = true;
    pass &= dut.concat_indices == expected_concat;
    pass &= dut.cluster_index == 2;
    pass &= static_cast<std::int16_t>(dut.maximum_value) == 12;
    pass &= static_cast<std::int16_t>(dut.minimum_value) == -7;
    pass &= dut.maximum_index == 2;  // tie at lanes 2/3 chooses the left lane.
    pass &= dut.minimum_index == 1;
    pass &= dut.histogram_valid == 1;
    pass &= dut.histogram == 0x249;  // four bins, each count is one.
    if (!pass) {
        std::cerr << "OASIS RTL smoke failure: concat=" << dut.concat_indices
                  << " cluster=" << static_cast<int>(dut.cluster_index)
                  << " max=" << static_cast<std::int16_t>(dut.maximum_value)
                  << " min=" << static_cast<std::int16_t>(dut.minimum_value)
                  << " histogram=" << dut.histogram << "\n";
        return 1;
    }
    std::cout << "OASIS RTL smoke PASS\n";
    return 0;
}
