`timescale 1ns/1ps

// Fully unrolled concatenation plane for one reported OASIS PE Line. Histogram
// counters consume 16-index slices through oasis_index_counter instances.
module oasis_pe_line #(
    parameter integer CONCAT_UNITS = 4096,
    parameter integer ACTIVATION_BITS = 4,
    parameter integer WEIGHT_BITS = 4
) (
    input  wire [CONCAT_UNITS*ACTIVATION_BITS-1:0] activation_indices,
    input  wire [CONCAT_UNITS*WEIGHT_BITS-1:0]     weight_indices,
    output wire [CONCAT_UNITS*(ACTIVATION_BITS+WEIGHT_BITS)-1:0] concat_indices
);
    genvar unit_index;
    generate
        for (unit_index = 0; unit_index < CONCAT_UNITS; unit_index = unit_index + 1) begin : gen_concat
            oasis_concat #(
                .ACTIVATION_BITS(ACTIVATION_BITS),
                .WEIGHT_BITS(WEIGHT_BITS)
            ) concat_unit (
                .activation_index(activation_indices[unit_index*ACTIVATION_BITS +: ACTIVATION_BITS]),
                .weight_index(weight_indices[unit_index*WEIGHT_BITS +: WEIGHT_BITS]),
                .concat_index(concat_indices[unit_index*(ACTIVATION_BITS+WEIGHT_BITS) +: (ACTIVATION_BITS+WEIGHT_BITS)])
            );
        end
    endgenerate
endmodule
