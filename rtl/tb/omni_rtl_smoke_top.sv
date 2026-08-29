`timescale 1ns/1ps

module omni_rtl_smoke_top (
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  bqu_valid,
    input  wire signed [15:0]    bqu_value,
    input  wire [1:0]            bit_width_minus_two,
    input  wire [4:0]            scale_shift,
    output wire                  bqu_valid_out,
    output wire [3:0]            quantized_index,
    output wire signed [4:0]     signed_code,
    input  wire                  lgu_valid,
    input  wire signed [15:0]    activation,
    input  wire signed [15:0]    centroid,
    input  wire signed [15:0]    correction,
    output wire                  lgu_valid_out,
    output wire signed [33:0]    lut_entry,
    input  wire                  lut_write_enable,
    input  wire [3:0]            lut_write_address,
    input  wire signed [33:0]    lut_write_data,
    input  wire                  clear_accumulator,
    input  wire                  lookup_valid,
    input  wire [3:0]            lookup_index,
    output wire signed [47:0]    accumulator
);
    omni_bqu bqu (
        .clk(clk),
        .reset(reset),
        .valid_in(bqu_valid),
        .value(bqu_value),
        .bit_width_minus_two(bit_width_minus_two),
        .scale_shift(scale_shift),
        .valid_out(bqu_valid_out),
        .quantized_index(quantized_index),
        .signed_code(signed_code)
    );

    omni_lgu lgu (
        .clk(clk),
        .reset(reset),
        .valid_in(lgu_valid),
        .activation(activation),
        .centroid(centroid),
        .correction(correction),
        .valid_out(lgu_valid_out),
        .lut_entry(lut_entry)
    );

    omni_lut_pe pe (
        .clk(clk),
        .reset(reset),
        .lut_write_enable(lut_write_enable),
        .lut_write_address(lut_write_address),
        .lut_write_data(lut_write_data),
        .clear_accumulator(clear_accumulator),
        .lookup_valid(lookup_valid),
        .lookup_index(lookup_index),
        .accumulator(accumulator)
    );
endmodule
