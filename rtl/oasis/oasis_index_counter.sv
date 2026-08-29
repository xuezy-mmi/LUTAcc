`timescale 1ns/1ps

// One paper-style N-input counter: decode all indices to one-hot vectors and
// sum every row. OASIS instantiates 32 16-input counters per PE Line.
module oasis_index_counter #(
    parameter integer INDEX_BITS = 8,
    parameter integer LANES = 16,
    parameter integer COUNT_BITS = 5,
    parameter integer LEVELS = (1 << INDEX_BITS)
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              valid_in,
    input  wire [LANES*INDEX_BITS-1:0]       indices,
    output reg                               valid_out,
    output reg  [LEVELS*COUNT_BITS-1:0]      histogram
);
    integer bin_index;
    integer lane_index;
    integer count_value;
    reg [LEVELS*COUNT_BITS-1:0] histogram_comb;

    always @* begin
        histogram_comb = {LEVELS*COUNT_BITS{1'b0}};
        for (bin_index = 0; bin_index < LEVELS; bin_index = bin_index + 1) begin
            count_value = 0;
            for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
                if (indices[lane_index*INDEX_BITS +: INDEX_BITS] == bin_index[INDEX_BITS-1:0])
                    count_value = count_value + 1;
            end
            histogram_comb[bin_index*COUNT_BITS +: COUNT_BITS] = count_value[COUNT_BITS-1:0];
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            valid_out <= 1'b0;
            histogram <= {LEVELS*COUNT_BITS{1'b0}};
        end else begin
            valid_out <= valid_in;
            if (valid_in)
                histogram <= histogram_comb;
        end
    end
endmodule
