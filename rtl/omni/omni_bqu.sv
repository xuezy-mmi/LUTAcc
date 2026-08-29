`timescale 1ns/1ps

// Bit Quantization Unit (BQU): variable 2/3/4-bit symmetric quantization.
// scale_shift is the fixed-point power-of-two scale selected by the compiler.
module omni_bqu #(
    parameter integer VALUE_BITS = 16
) (
    input  wire                          clk,
    input  wire                          reset,
    input  wire                          valid_in,
    input  wire signed [VALUE_BITS-1:0]  value,
    input  wire [1:0]                    bit_width_minus_two,
    input  wire [4:0]                    scale_shift,
    output reg                           valid_out,
    output reg  [3:0]                    quantized_index,
    output reg  signed [4:0]             signed_code
);
    reg signed [VALUE_BITS-1:0] shifted;
    reg signed [4:0] qmin;
    reg signed [4:0] qmax;
    reg signed [4:0] clipped;
    reg signed [VALUE_BITS-1:0] qmin_extended;
    reg signed [VALUE_BITS-1:0] qmax_extended;
    reg [3:0] index_comb;

    always @* begin
        shifted = $signed(value) >>> scale_shift;
        case (bit_width_minus_two)
            2'd0: begin qmin = -5'sd2; qmax = 5'sd1; end
            2'd1: begin qmin = -5'sd4; qmax = 5'sd3; end
            default: begin qmin = -5'sd8; qmax = 5'sd7; end
        endcase
        qmin_extended = {{(VALUE_BITS-5){qmin[4]}}, qmin};
        qmax_extended = {{(VALUE_BITS-5){qmax[4]}}, qmax};
        if ($signed(shifted) < $signed(qmin_extended))
            clipped = qmin;
        else if ($signed(shifted) > $signed(qmax_extended))
            clipped = qmax;
        else
            clipped = shifted[4:0];
        index_comb = clipped[3:0] - qmin[3:0];
    end

    always @(posedge clk) begin
        if (reset) begin
            valid_out <= 1'b0;
            quantized_index <= 4'd0;
            signed_code <= 5'sd0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                quantized_index <= index_comb;
                signed_code <= clipped;
            end
        end
    end
endmodule
