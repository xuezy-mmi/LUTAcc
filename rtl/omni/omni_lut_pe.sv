`timescale 1ns/1ps

module omni_lut_pe #(
    parameter integer LUT_ENTRIES = 16,
    parameter integer INDEX_BITS = 4,
    parameter integer ENTRY_BITS = 34,
    parameter integer ACC_BITS = 48
) (
    input  wire                          clk,
    input  wire                          reset,
    input  wire                          lut_write_enable,
    input  wire [INDEX_BITS-1:0]         lut_write_address,
    input  wire signed [ENTRY_BITS-1:0]  lut_write_data,
    input  wire                          clear_accumulator,
    input  wire                          lookup_valid,
    input  wire [INDEX_BITS-1:0]         lookup_index,
    output reg signed [ACC_BITS-1:0]     accumulator
);
    reg signed [ENTRY_BITS-1:0] lut [0:LUT_ENTRIES-1];
    integer entry_index;

    always @(posedge clk) begin
        if (reset) begin
            accumulator <= {ACC_BITS{1'b0}};
            for (entry_index = 0; entry_index < LUT_ENTRIES; entry_index = entry_index + 1)
                lut[entry_index] <= {ENTRY_BITS{1'b0}};
        end else begin
            if (lut_write_enable)
                lut[lut_write_address] <= lut_write_data;
            if (clear_accumulator)
                accumulator <= {ACC_BITS{1'b0}};
            else if (lookup_valid)
                accumulator <= accumulator + $signed({
                    {(ACC_BITS-ENTRY_BITS){lut[lookup_index][ENTRY_BITS-1]}},
                    lut[lookup_index]
                });
        end
    end
endmodule
