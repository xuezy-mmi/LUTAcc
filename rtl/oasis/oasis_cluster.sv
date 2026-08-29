`timescale 1ns/1ps

// Fixed-point binary-boundary equivalent of nearest-centroid assignment.
// Boundaries must be sorted; equality selects the centroid on the right,
// matching numpy.searchsorted(..., side="right") in the functional model.
module oasis_cluster #(
    parameter integer VALUE_BITS = 16,
    parameter integer CENTROIDS = 16,
    parameter integer INDEX_BITS = 4
) (
    input  wire signed [VALUE_BITS-1:0] value,
    input  wire [(CENTROIDS-1)*VALUE_BITS-1:0] boundaries,
    output reg  [INDEX_BITS-1:0] cluster_index
);
    integer boundary_index;
    reg signed [VALUE_BITS-1:0] boundary_value;

    always @* begin
        cluster_index = {INDEX_BITS{1'b0}};
        boundary_value = {VALUE_BITS{1'b0}};
        for (boundary_index = 0; boundary_index < CENTROIDS-1; boundary_index = boundary_index + 1) begin
            boundary_value = boundaries[boundary_index*VALUE_BITS +: VALUE_BITS];
            if ($signed(value) >= $signed(boundary_value)) begin
                cluster_index = boundary_index[INDEX_BITS-1:0];
                cluster_index = cluster_index + {{(INDEX_BITS-1){1'b0}}, 1'b1};
            end
        end
    end
endmodule
