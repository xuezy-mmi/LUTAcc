`timescale 1ns/1ps

module oasis_compensation_mac #(
    parameter integer LANES = 8,
    parameter integer VALUE_BITS = 16,
    parameter integer ACC_BITS = 40
) (
    input  wire                             clk,
    input  wire                             reset,
    input  wire                             clear,
    input  wire                             valid_in,
    input  wire [LANES*VALUE_BITS-1:0]      residuals,
    input  wire [LANES*VALUE_BITS-1:0]      weights,
    output reg signed [ACC_BITS-1:0]        accumulator
);
    integer lane_index;
    reg signed [ACC_BITS-1:0] dot_product;
    reg signed [VALUE_BITS-1:0] residual_item;
    reg signed [VALUE_BITS-1:0] weight_item;

    always @* begin
        dot_product = {ACC_BITS{1'b0}};
        residual_item = {VALUE_BITS{1'b0}};
        weight_item = {VALUE_BITS{1'b0}};
        for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
            residual_item = residuals[lane_index*VALUE_BITS +: VALUE_BITS];
            weight_item = weights[lane_index*VALUE_BITS +: VALUE_BITS];
            dot_product = dot_product + $signed(residual_item) * $signed(weight_item);
        end
    end

    always @(posedge clk) begin
        if (reset || clear)
            accumulator <= {ACC_BITS{1'b0}};
        else if (valid_in)
            accumulator <= accumulator + dot_product;
    end
endmodule
