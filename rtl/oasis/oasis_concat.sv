`timescale 1ns/1ps

module oasis_concat #(
    parameter integer ACTIVATION_BITS = 4,
    parameter integer WEIGHT_BITS = 4
) (
    input  wire [ACTIVATION_BITS-1:0] activation_index,
    input  wire [WEIGHT_BITS-1:0]     weight_index,
    output wire [ACTIVATION_BITS+WEIGHT_BITS-1:0] concat_index
);
    assign concat_index = {activation_index, weight_index};
endmodule
