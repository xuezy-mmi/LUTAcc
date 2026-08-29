`timescale 1ns/1ps

// Reported 32x4 logical LUT-PE array. The load bus is broadcast; each PE gets
// its own lookup index and output-stationary accumulator.
module omni_array #(
    parameter integer ROWS = 32,
    parameter integer COLS = 4,
    parameter integer PES = ROWS * COLS,
    parameter integer INDEX_BITS = 4,
    parameter integer ENTRY_BITS = 34,
    parameter integer ACC_BITS = 48
) (
    input  wire                          clk,
    input  wire                          reset,
    input  wire                          lut_write_enable,
    input  wire [INDEX_BITS-1:0]         lut_write_address,
    input  wire signed [ENTRY_BITS-1:0]  lut_write_data,
    input  wire                          clear_accumulators,
    input  wire                          lookup_valid,
    input  wire [PES*INDEX_BITS-1:0]     lookup_indices,
    output wire [PES*ACC_BITS-1:0]       accumulators
);
    genvar pe_index;
    generate
        for (pe_index = 0; pe_index < PES; pe_index = pe_index + 1) begin : gen_pe
            omni_lut_pe #(
                .LUT_ENTRIES(1 << INDEX_BITS),
                .INDEX_BITS(INDEX_BITS),
                .ENTRY_BITS(ENTRY_BITS),
                .ACC_BITS(ACC_BITS)
            ) pe (
                .clk(clk),
                .reset(reset),
                .lut_write_enable(lut_write_enable),
                .lut_write_address(lut_write_address),
                .lut_write_data(lut_write_data),
                .clear_accumulator(clear_accumulators),
                .lookup_valid(lookup_valid),
                .lookup_index(lookup_indices[pe_index*INDEX_BITS +: INDEX_BITS]),
                .accumulator(accumulators[pe_index*ACC_BITS +: ACC_BITS])
            );
        end
    endgenerate
endmodule
