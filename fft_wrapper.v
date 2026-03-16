// ============================================================
// fft_wrapper.v
// Wraps Altera FFT MegaCore IP
// COE838 — FMCW Radar Range-Doppler SoC
//
// The Altera FFT IP uses Avalon-ST (streaming) interface:
//   - Sink  (input):  we push samples IN  to the FFT
//   - Source (output): we pull results OUT of the FFT
//
// HOW TO CONFIGURE ALTERA FFT IP IN QSYS/IP CATALOG:
//   Transform Length:   256
//   Data Width:         16 bits (I and Q each 16-bit)
//   Twiddle Width:      16 bits
//   Engine Architecture: Streaming (variable)
//   Direction:          Forward only (or run-time selectable)
//   Output Order:       Natural order (bit-reversed input, natural output)
//
// The IP component name you instantiate below should match
// exactly what Quartus generates — check fft_256.v after IP gen.
// ============================================================

module fft_wrapper (
    input  wire        clk,
    input  wire        reset,

    // ---- From accelerator FSM (sink side — feeding the FFT) ----
    input  wire        sink_valid,
    input  wire [31:0] sink_data,    // [31:16]=Im [15:0]=Re (int16)
    input  wire        sink_sop,
    input  wire        sink_eop,
    output wire        sink_ready,

    // ---- To accelerator FSM (source side — reading FFT output) ----
    output wire        source_valid,
    output wire [31:0] source_data,  // [31:16]=Im [15:0]=Re (int16)
    output wire        source_sop,
    output wire        source_eop,
    input  wire        source_ready,

    // ---- Direction ----
    input  wire        inverse        // 0=forward, 1=inverse
);

// ============================================================
// Altera FFT IP instantiation
//
// IMPORTANT: The port names below are the STANDARD names that
// Altera's FFT MegaCore generates. If yours differ slightly,
// update to match what the IP wizard produced.
//
// After generating the IP, open the produced .v file and verify:
//   - clk, reset_n
//   - sink_valid, sink_ready, sink_sop, sink_eop
//   - sink_real, sink_imag (separate real/imag ports)
//   - source_valid, source_ready, source_sop, source_eop  
//   - source_real, source_imag
// ============================================================

// Split our packed 32-bit data into separate Re/Im for the IP
wire signed [15:0] sink_re = sink_data[15:0];
wire signed [15:0] sink_im = sink_data[31:16];

wire signed [15:0] source_re_raw;
wire signed [15:0] source_im_raw;

// Pack IP outputs back into 32-bit word for FSM
assign source_data = {source_im_raw, source_re_raw};

// Active-low reset for Altera IP
wire reset_n = ~reset;

// ---- Instantiate Altera FFT IP ----
// Replace "fft_256" with the exact name you gave your IP in the wizard
fft_256 fft_inst (
    .clk          (clk),
    .reset_n      (reset_n),

    // Sink (input) — samples going INTO the FFT
    .sink_valid   (sink_valid),
    .sink_ready   (sink_ready),
    .sink_error   (2'b00),          // no upstream error
    .sink_sop     (sink_sop),
    .sink_eop     (sink_eop),
    .sink_real    (sink_re),
    .sink_imag    (sink_im),

    // Source (output) — bins coming OUT of the FFT
    .source_valid (source_valid),
    .source_ready (source_ready),
    .source_error (),               // ignore error output
    .source_sop   (source_sop),
    .source_eop   (source_eop),
    .source_real  (source_re_raw),
    .source_imag  (source_im_raw),

    // Direction select
    .inverse      (inverse)
);

endmodule
