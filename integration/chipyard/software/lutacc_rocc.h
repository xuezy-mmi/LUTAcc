#ifndef LUTACC_ROCC_H
#define LUTACC_ROCC_H

#include <stdint.h>

static inline uint64_t lutacc_insn(uint64_t rs1, uint64_t rs2, unsigned funct) {
    uint64_t rd;
    switch (funct) {
    case 0:
        __asm__ volatile (".insn r 0x7b, 0, 0, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2));
        break;
    case 1:
        __asm__ volatile (".insn r 0x7b, 0, 1, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2));
        break;
    case 2:
        __asm__ volatile (".insn r 0x7b, 0, 2, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2));
        break;
    default:
        __asm__ volatile (".insn r 0x7b, 0, 3, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2));
        break;
    }
    return rd;
}

static inline void lutacc_configure(
    uint32_t m, uint32_t k, uint32_t n, uint32_t activation_bits,
    uint32_t outlier_basis_points) {
    const uint64_t rs1 = ((uint64_t)k << 32) | m;
    const uint64_t rs2 = ((uint64_t)outlier_basis_points << 34)
                       | ((uint64_t)(activation_bits - 2) << 32) | n;
    (void)lutacc_insn(rs1, rs2, 0);
}

#endif
