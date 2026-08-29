`timescale 1ns/1ps

// Leaf primitive for the two-fold Orizuru tree. Strict comparisons retain the
// lowest lane index on FP16/fixed-point ties, as required by the paper.
module orizuru_tree #(
    parameter integer LANES = 16,
    parameter integer VALUE_BITS = 16,
    parameter integer INDEX_BITS = 4
) (
    input  wire [LANES*VALUE_BITS-1:0] values,
    output reg  signed [VALUE_BITS-1:0] maximum_value,
    output reg  signed [VALUE_BITS-1:0] minimum_value,
    output reg  [INDEX_BITS-1:0] maximum_index,
    output reg  [INDEX_BITS-1:0] minimum_index
);
    integer lane_index;
    reg signed [VALUE_BITS-1:0] lane_value;

    always @* begin
        maximum_value = values[0 +: VALUE_BITS];
        minimum_value = values[0 +: VALUE_BITS];
        maximum_index = {INDEX_BITS{1'b0}};
        minimum_index = {INDEX_BITS{1'b0}};
        lane_value = {VALUE_BITS{1'b0}};
        for (lane_index = 1; lane_index < LANES; lane_index = lane_index + 1) begin
            lane_value = values[lane_index*VALUE_BITS +: VALUE_BITS];
            if ($signed(lane_value) > $signed(maximum_value)) begin
                maximum_value = lane_value;
                maximum_index = lane_index[INDEX_BITS-1:0];
            end
            if ($signed(lane_value) < $signed(minimum_value)) begin
                minimum_value = lane_value;
                minimum_index = lane_index[INDEX_BITS-1:0];
            end
        end
    end
endmodule
