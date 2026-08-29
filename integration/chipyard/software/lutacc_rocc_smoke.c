#include "lutacc_rocc.h"

int main(void) {
    lutacc_configure(1, 4096, 4096, 4, 100);
    const uint64_t oasis_cycles = lutacc_insn(0, 0, 1);
    lutacc_configure(1, 128, 2048, 4, 0);
    const uint64_t omni_aw_cycles = lutacc_insn(0, 0, 2);
    return oasis_cycles == 2560 && omni_aw_cycles == 17952 ? 0 : 1;
}
