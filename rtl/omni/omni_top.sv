`timescale 1ns/1ps

module omni_top #(
    parameter integer ROWS = 32,
    parameter integer COLS = 4,
    parameter integer PES = ROWS * COLS,
    parameter integer INDEX_BITS = 4,
    parameter integer ENTRY_BITS = 34,
    parameter integer ACC_BITS = 48
) (
    input  wire                          clk,
    input  wire                          reset,
    input  wire                          start,
    input  wire                          decode_phase,
    input  wire                          bqu_done,
    input  wire                          lgu_done,
    input  wire                          array_done,
    input  wire                          lut_write_enable,
    input  wire [INDEX_BITS-1:0]         lut_write_address,
    input  wire signed [ENTRY_BITS-1:0]  lut_write_data,
    input  wire                          clear_accumulators,
    input  wire                          lookup_valid,
    input  wire [PES*INDEX_BITS-1:0]     lookup_indices,
    output wire [PES*ACC_BITS-1:0]       accumulators,
    output wire                          weight_stationary,
    output wire                          value_output_stationary,
    output wire                          done
);
    wire bqu_start_unused;
    wire lgu_start_unused;
    wire array_start_unused;

    omni_controller controller (
        .clk(clk),
        .reset(reset),
        .start(start),
        .decode_phase(decode_phase),
        .bqu_done(bqu_done),
        .lgu_done(lgu_done),
        .array_done(array_done),
        .bqu_start(bqu_start_unused),
        .lgu_start(lgu_start_unused),
        .array_start(array_start_unused),
        .weight_stationary(weight_stationary),
        .value_output_stationary(value_output_stationary),
        .done(done)
    );

    omni_array #(
        .ROWS(ROWS),
        .COLS(COLS),
        .PES(PES),
        .INDEX_BITS(INDEX_BITS),
        .ENTRY_BITS(ENTRY_BITS),
        .ACC_BITS(ACC_BITS)
    ) array (
        .clk(clk),
        .reset(reset),
        .lut_write_enable(lut_write_enable),
        .lut_write_address(lut_write_address),
        .lut_write_data(lut_write_data),
        .clear_accumulators(clear_accumulators),
        .lookup_valid(lookup_valid),
        .lookup_indices(lookup_indices),
        .accumulators(accumulators)
    );
endmodule
