`timescale 1ns/1ps

// LUT Generation Unit. The correction term is learned offline/online by BQU
// calibration and folded into the product at LUT creation time.
module omni_lgu #(
    parameter integer VALUE_BITS = 16,
    parameter integer PRODUCT_BITS = 34
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              valid_in,
    input  wire signed [VALUE_BITS-1:0]      activation,
    input  wire signed [VALUE_BITS-1:0]      centroid,
    input  wire signed [VALUE_BITS-1:0]      correction,
    output reg                               valid_out,
    output reg signed [PRODUCT_BITS-1:0]     lut_entry
);
    reg signed [VALUE_BITS:0] corrected_centroid;
    reg signed [PRODUCT_BITS-1:0] product_comb;

    always @* begin
        corrected_centroid = $signed(centroid) + $signed(correction);
        product_comb = $signed(activation) * $signed(corrected_centroid);
    end

    always @(posedge clk) begin
        if (reset) begin
            valid_out <= 1'b0;
            lut_entry <= {PRODUCT_BITS{1'b0}};
        end else begin
            valid_out <= valid_in;
            if (valid_in)
                lut_entry <= product_comb;
        end
    end
endmodule
