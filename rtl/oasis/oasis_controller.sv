`timescale 1ns/1ps

// Branch-join controller. Main and outlier branches may finish in either order;
// merge starts only after both completion tokens are observed.
module oasis_controller (
    input  wire clk,
    input  wire reset,
    input  wire start,
    input  wire main_done,
    input  wire outlier_done,
    input  wire merge_done,
    output reg  main_start,
    output reg  outlier_start,
    output reg  merge_start,
    output reg  done
);
    localparam [1:0] IDLE = 2'd0;
    localparam [1:0] RUN_BRANCHES = 2'd1;
    localparam [1:0] RUN_MERGE = 2'd2;
    reg [1:0] state;
    reg main_seen;
    reg outlier_seen;

    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            main_seen <= 1'b0;
            outlier_seen <= 1'b0;
            main_start <= 1'b0;
            outlier_start <= 1'b0;
            merge_start <= 1'b0;
            done <= 1'b0;
        end else begin
            main_start <= 1'b0;
            outlier_start <= 1'b0;
            merge_start <= 1'b0;
            done <= 1'b0;
            case (state)
                IDLE: begin
                    main_seen <= 1'b0;
                    outlier_seen <= 1'b0;
                    if (start) begin
                        main_start <= 1'b1;
                        outlier_start <= 1'b1;
                        state <= RUN_BRANCHES;
                    end
                end
                RUN_BRANCHES: begin
                    if (main_done)
                        main_seen <= 1'b1;
                    if (outlier_done)
                        outlier_seen <= 1'b1;
                    if ((main_seen || main_done) && (outlier_seen || outlier_done)) begin
                        merge_start <= 1'b1;
                        state <= RUN_MERGE;
                    end
                end
                RUN_MERGE: begin
                    if (merge_done) begin
                        done <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
