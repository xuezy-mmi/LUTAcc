#include "VLUTAccTimingBlackBox.h"
#include "verilated.h"

#include <cstdint>
#include <iostream>

static void tick(VLUTAccTimingBlackBox &dut) {
    dut.clock = 0;
    dut.eval();
    dut.clock = 1;
    dut.eval();
}

static std::uint64_t launch(
    VLUTAccTimingBlackBox &dut,
    bool omni,
    std::uint32_t m,
    std::uint32_t k,
    std::uint32_t n,
    std::uint32_t activation_bits,
    std::uint32_t outlier_basis_points) {
    dut.designOmni = omni;
    dut.m = m;
    dut.k = k;
    dut.n = n;
    dut.activationBits = activation_bits;
    dut.outlierBasisPoints = outlier_basis_points;
    dut.start = 1;
    tick(dut);
    dut.start = 0;
    for (int timeout = 0; timeout < 100000; ++timeout) {
        tick(dut);
        if (dut.done)
            return dut.cycles;
    }
    return 0;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    VLUTAccTimingBlackBox dut;
    dut.reset = 1;
    dut.start = 0;
    tick(dut);
    dut.reset = 0;
    const std::uint64_t oasis = launch(dut, false, 1, 4096, 4096, 4, 100);
    const std::uint64_t oasis_a3 = launch(dut, false, 1, 4096, 4096, 3, 100);
    const std::uint64_t oasis_orizuru = launch(dut, false, 1, 128, 64, 4, 500);
    const std::uint64_t oasis_batched = launch(dut, false, 8, 512, 256, 4, 100);
    const std::uint64_t omni = launch(dut, true, 1, 128, 2048, 4, 0);
    const std::uint64_t omni_small = launch(dut, true, 2, 64, 96, 4, 0);
    if (oasis != 2560 || oasis_a3 != 2304 || oasis_orizuru != 126
        || oasis_batched != 2176 || omni != 17952 || omni_small != 872) {
        std::cerr << "Chipyard timing bridge failure: oasis=" << oasis
                  << " oasis_a3=" << oasis_a3
                  << " oasis_orizuru=" << oasis_orizuru
                  << " oasis_batched=" << oasis_batched
                  << " omni=" << omni << " omni_small=" << omni_small << "\n";
        return 1;
    }
    std::cout << "Chipyard timing bridge PASS: oasis=" << oasis
              << " oasis_a3=" << oasis_a3 << " omni_aw=" << omni << "\n";
    return 0;
}
