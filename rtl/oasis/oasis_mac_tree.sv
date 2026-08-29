`timescale 1ns/1ps

// Synthesizable fixed-point stand-in for the paper's 32-input FP16 MAC tree.
// The control/reduction topology is exact; replace the signed multiplier with a
// technology FP16 macro for ASIC PPA runs.
module oasis_mac_tree #(
    parameter integer INPUTS = 32,
    parameter integer COUNT_BITS = 13,
    parameter integer VALUE_BITS = 16,
    parameter integer ACC_BITS = 40
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              valid_in,
    input  wire [INPUTS*COUNT_BITS-1:0]      counts,
    input  wire [INPUTS*VALUE_BITS-1:0]      lut_values,
    output reg                               valid_out,
    output reg signed [ACC_BITS-1:0]         sum
);
    integer input_index;
    reg signed [ACC_BITS-1:0] sum_comb;
    reg signed [VALUE_BITS-1:0] value_item;
    reg [COUNT_BITS-1:0] count_item;

    always @* begin
        sum_comb = {ACC_BITS{1'b0}};
        value_item = {VALUE_BITS{1'b0}};
        count_item = {COUNT_BITS{1'b0}};
        for (input_index = 0; input_index < INPUTS; input_index = input_index + 1) begin
            value_item = lut_values[input_index*VALUE_BITS +: VALUE_BITS];
            count_item = counts[input_index*COUNT_BITS +: COUNT_BITS];
            sum_comb = sum_comb + $signed(value_item) * $signed({1'b0, count_item});
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            valid_out <= 1'b0;
            sum <= {ACC_BITS{1'b0}};
        end else begin
            valid_out <= valid_in;
            if (valid_in)
                sum <= sum_comb;
        end
    end
endmodule
