`timescale 1ns/1ps

module oasis_rtl_smoke_top (
    input  wire         clk,
    input  wire         reset,
    input  wire         valid_in,
    input  wire [7:0]   activation_indices,
    input  wire [7:0]   weight_indices,
    input  wire signed [15:0] sample,
    input  wire [47:0]  boundaries,
    input  wire [63:0]  values,
    output wire [15:0]  concat_indices,
    output wire [1:0]   cluster_index,
    output wire signed [15:0] maximum_value,
    output wire signed [15:0] minimum_value,
    output wire [1:0]   maximum_index,
    output wire [1:0]   minimum_index,
    output wire         histogram_valid,
    output wire [11:0]  histogram
);
    genvar lane;
    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin : smoke_concat
            oasis_concat #(.ACTIVATION_BITS(2), .WEIGHT_BITS(2)) concat_unit (
                .activation_index(activation_indices[lane*2 +: 2]),
                .weight_index(weight_indices[lane*2 +: 2]),
                .concat_index(concat_indices[lane*4 +: 4])
            );
        end
    endgenerate

    oasis_cluster #(.VALUE_BITS(16), .CENTROIDS(4), .INDEX_BITS(2)) cluster_unit (
        .value(sample),
        .boundaries(boundaries),
        .cluster_index(cluster_index)
    );

    orizuru_tree #(.LANES(4), .VALUE_BITS(16), .INDEX_BITS(2)) orizuru_unit (
        .values(values),
        .maximum_value(maximum_value),
        .minimum_value(minimum_value),
        .maximum_index(maximum_index),
        .minimum_index(minimum_index)
    );

    oasis_index_counter #(.INDEX_BITS(2), .LANES(4), .COUNT_BITS(3), .LEVELS(4)) counter_unit (
        .clk(clk),
        .reset(reset),
        .valid_in(valid_in),
        .indices(activation_indices),
        .valid_out(histogram_valid),
        .histogram(histogram)
    );
endmodule
