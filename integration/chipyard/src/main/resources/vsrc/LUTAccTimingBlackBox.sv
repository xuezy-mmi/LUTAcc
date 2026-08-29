`timescale 1ns/1ps

module LUTAccTimingBlackBox (
    input  wire        clock,
    input  wire        reset,
    input  wire        start,
    input  wire        designOmni,
    input  wire [31:0] m,
    input  wire [31:0] k,
    input  wire [31:0] n,
    input  wire [2:0]  activationBits,
    input  wire [15:0] outlierBasisPoints,
    output reg         busy,
    output reg         done,
    output reg [63:0]  cycles
);
    function automatic [63:0] ceil_div;
        input [63:0] numerator;
        input [63:0] denominator;
        begin
            ceil_div = (numerator + denominator - 1) / denominator;
        end
    endfunction

    function automatic [63:0] ceil_log2;
        input [63:0] value;
        reg [63:0] shifted;
        integer bit_index;
        begin
            shifted = value - 1;
            ceil_log2 = 0;
            for (bit_index = 0; bit_index < 64; bit_index = bit_index + 1) begin
                if (shifted != 0) begin
                    ceil_log2 = ceil_log2 + 1;
                    shifted = shifted >> 1;
                end
            end
        end
    endfunction

    reg [63:0] remaining;
    reg [63:0] m64;
    reg [63:0] k64;
    reg [63:0] n64;
    reg [63:0] output_waves;
    reg [63:0] reduction_tiles;
    reg [63:0] activation_quant;
    reg [63:0] concat_cycles;
    reg [63:0] index_distribution;
    reg [63:0] reduction;
    reg [63:0] merge;
    reg [63:0] main_branch;
    reg [63:0] k_each;
    reg [63:0] outliers;
    reg [63:0] tree_levels;
    reg [63:0] units_per_token;
    reg [63:0] token_parallelism;
    reg [63:0] detector_token_waves;
    reg [63:0] detector_per_wave;
    reg [63:0] outlier_detection;
    reg [63:0] outlier_branch;
    reg [63:0] compensation;
    reg [63:0] modeled_cycles;
    reg [63:0] lut_entries;
    reg [63:0] dram_cycles;
    reg [63:0] omni_lgu_entries;
    reg [63:0] omni_lgu_cycles;
    reg [63:0] omni_lookups;
    reg [63:0] omni_token_tiles;
    reg [63:0] omni_output_tiles;
    reg [63:0] omni_array_fill;
    reg [63:0] omni_array_cycles;
    reg [63:0] omni_core_cycles;
    reg [63:0] omni_dram_cycles;

    always @* begin
        m64 = {32'd0, m};
        k64 = {32'd0, k};
        n64 = {32'd0, n};
        output_waves = ceil_div(m64 * n64, 64'd16);
        reduction_tiles = ceil_div(k64, 64'd4096);
        activation_quant = ceil_div(m64 * k64, 64'd2);
        concat_cycles = output_waves * reduction_tiles;
        index_distribution = output_waves * ceil_div(k64, 64'd1024);
        lut_entries = (activationBits == 3) ? 128 : 256;
        reduction = output_waves * (ceil_div(lut_entries, 64'd32) + 1);
        merge = output_waves;
        main_branch = activation_quant;
        if (concat_cycles > main_branch)
            main_branch = concat_cycles;
        if (index_distribution > main_branch)
            main_branch = index_distribution;
        if (reduction > main_branch)
            main_branch = reduction;
        k_each = (k64 * {48'd0, outlierBasisPoints}) / 20000;
        outliers = 2 * k_each;
        tree_levels = ceil_log2(k64);
        if (tree_levels == 0)
            tree_levels = 1;
        units_per_token = ceil_div(k64, 64'd16);
        token_parallelism = 273 / units_per_token;
        if (token_parallelism == 0)
            token_parallelism = 1;
        detector_token_waves = ceil_div(m64, token_parallelism);
        detector_per_wave = units_per_token * 3
                          + 2 * k_each * tree_levels
                          + 8 * tree_levels;
        outlier_detection = detector_token_waves * detector_per_wave;
        compensation = output_waves * (ceil_div(outliers, 64'd8) + 1);
        outlier_branch = (outlier_detection > compensation)
                       ? outlier_detection : compensation;
        modeled_cycles = ((main_branch > outlier_branch)
                         ? main_branch : outlier_branch) + merge;
        dram_cycles = ceil_div(m64 * k64 * 2 + m64 * n64 * 2, 64'd128);
        if (dram_cycles > modeled_cycles)
            modeled_cycles = dram_cycles;

        // funct=2 is the compiler's Omni-LUT activation-weight GEMM path.
        // Constants match configs/hardware/omni_lut_inferred.yaml.
        omni_lgu_entries = m64 * k64 * 16;
        omni_lgu_cycles = ceil_div(omni_lgu_entries, 64'd64);
        omni_lookups = m64 * k64 * n64;
        omni_token_tiles = ceil_div(m64, 64'd32);
        omni_output_tiles = ceil_div(n64, 64'd4);
        omni_array_fill = omni_token_tiles * omni_output_tiles * 34;
        omni_array_cycles = ceil_div(omni_lookups, 64'd512) + omni_array_fill;
        omni_core_cycles = omni_lgu_cycles + omni_array_cycles;
        omni_dram_cycles = ceil_div(m64 * k64 * 2 + m64 * n64 * 2, 64'd128);
        if (designOmni)
            modeled_cycles = (omni_core_cycles > omni_dram_cycles)
                           ? omni_core_cycles : omni_dram_cycles;
    end

    always @(posedge clock) begin
        if (reset) begin
            busy <= 1'b0;
            done <= 1'b0;
            cycles <= 64'd0;
            remaining <= 64'd0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1;
                remaining <= modeled_cycles;
                cycles <= modeled_cycles;
            end else if (busy && remaining > 1) begin
                remaining <= remaining - 1;
            end else if (busy) begin
                remaining <= 64'd0;
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end
endmodule
