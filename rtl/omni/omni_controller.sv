`timescale 1ns/1ps

module omni_controller (
    input  wire clk,
    input  wire reset,
    input  wire start,
    input  wire decode_phase,
    input  wire bqu_done,
    input  wire lgu_done,
    input  wire array_done,
    output reg  bqu_start,
    output reg  lgu_start,
    output reg  array_start,
    output reg  weight_stationary,
    output reg  value_output_stationary,
    output reg  done
);
    localparam [1:0] IDLE = 2'd0;
    localparam [1:0] BQU = 2'd1;
    localparam [1:0] LGU = 2'd2;
    localparam [1:0] ARRAY = 2'd3;
    reg [1:0] state;

    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            bqu_start <= 1'b0;
            lgu_start <= 1'b0;
            array_start <= 1'b0;
            weight_stationary <= 1'b0;
            value_output_stationary <= 1'b0;
            done <= 1'b0;
        end else begin
            bqu_start <= 1'b0;
            lgu_start <= 1'b0;
            array_start <= 1'b0;
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        weight_stationary <= ~decode_phase;
                        value_output_stationary <= decode_phase;
                        bqu_start <= 1'b1;
                        state <= BQU;
                    end
                end
                BQU: begin
                    if (bqu_done) begin
                        lgu_start <= 1'b1;
                        state <= LGU;
                    end
                end
                LGU: begin
                    if (lgu_done) begin
                        array_start <= 1'b1;
                        state <= ARRAY;
                    end
                end
                ARRAY: begin
                    if (array_done) begin
                        done <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
