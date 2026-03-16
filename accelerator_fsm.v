// ============================================================
// accelerator_fsm.v
// Master FSM — Radar Range-Doppler Accelerator
// COE838 — FMCW Radar Range-Doppler SoC
//
// What this does:
//   1. Waits for START from control slave
//   2. Reads one chirp (NR samples) from input SRAM
//   3. Feeds samples into FFT IP
//   4. Computes magnitude on FFT output
//   5. Writes magnitude results to output SRAM
//   6. Repeats for all ND chirps
//   7. Asserts DONE, latches cycle count
//
// Input SRAM layout:
//   Each sample = 32 bits: [31:16]=Im, [15:0]=Re  (signed int16)
//   Chirp m, sample n → address = m*NR + n
//
// Output SRAM layout:
//   Each result = 32 bits: Re^2 + Im^2 (uint32)
//   Chirp m, sample n → address = m*NR + n
// ============================================================

module accelerator_fsm (
    input  wire        clk,
    input  wire        reset,

    // ---- From control slave ----
    input  wire        ctrl_start,
    input  wire        ctrl_reset,
    input  wire [9:0]  cfg_nr,         // range FFT length (e.g. 256)
    input  wire [6:0]  cfg_nd,         // number of chirps (e.g. 64)

    // ---- To control slave ----
    output reg         status_busy,
    output reg         status_done,
    output reg  [31:0] cycle_count,

    // ---- Input SRAM port (Avalon MM Master — read side) ----
    output reg  [15:0] isram_addr,     // word address into input SRAM
    output reg         isram_rd,       // read enable
    input  wire [31:0] isram_rddata,   // [31:16]=Im [15:0]=Re

    // ---- Output SRAM port (Avalon MM Master — write side) ----
    output reg  [15:0] osram_addr,     // word address into output SRAM
    output reg         osram_wr,       // write enable
    output reg  [31:0] osram_wrdata,   // magnitude result

    // ---- FFT IP interface (Avalon-ST streaming) ----
    // Sink (input to FFT)
    output reg         fft_sink_valid,
    output reg  [31:0] fft_sink_data,  // [31:16]=Im [15:0]=Re
    output reg         fft_sink_sop,   // start of packet
    output reg         fft_sink_eop,   // end of packet
    input  wire        fft_sink_ready, // FFT can accept data

    // Source (output from FFT)
    input  wire        fft_source_valid,
    input  wire [31:0] fft_source_data, // [31:16]=Im [15:0]=Re
    input  wire        fft_source_sop,
    input  wire        fft_source_eop,
    output reg         fft_source_ready,

    // FFT inverse flag (0 = forward FFT)
    output wire        fft_inverse
);

// Forward FFT always
assign fft_inverse = 1'b0;

// ============================================================
// FSM State Encoding
// ============================================================
localparam S_IDLE       = 4'd0;  // waiting for START
localparam S_LOAD       = 4'd1;  // read sample from input SRAM
localparam S_LOAD_WAIT  = 4'd2;  // wait 1 cycle for SRAM read latency
localparam S_FEED_FFT   = 4'd3;  // push sample into FFT sink
localparam S_WAIT_FFT   = 4'd4;  // wait for FFT output to appear
localparam S_WRITE_MAG  = 4'd5;  // compute magnitude, write output SRAM
localparam S_NEXT_CHIRP = 4'd6;  // increment chirp counter
localparam S_DONE       = 4'd7;  // assert done, latch cycles

reg [3:0] state;

// ---- Counters ----
reg [9:0]  sample_cnt;   // which sample within current chirp (0..NR-1)
reg [6:0]  chirp_cnt;    // which chirp we are processing (0..ND-1)
reg [15:0] sram_base;    // base address of current chirp in SRAM
reg [31:0] timer;        // free-running cycle counter

// ---- FFT output capture ----
wire signed [15:0] fft_re = fft_source_data[15:0];
wire signed [15:0] fft_im = fft_source_data[31:16];

// ---- Magnitude (Re^2 + Im^2) — use 32-bit, saturate ----
wire [31:0] mag_sq;
wire [47:0] re_sq = {{16{fft_re[15]}}, fft_re} * {{16{fft_re[15]}}, fft_re};
wire [47:0] im_sq = {{16{fft_im[15]}}, fft_im} * {{16{fft_im[15]}}, fft_im};
wire [48:0] mag_full = {1'b0, re_sq} + {1'b0, im_sq};
// Saturate to 32 bits if overflow
assign mag_sq = (|mag_full[48:32]) ? 32'hFFFFFFFF : mag_full[31:0];

// ============================================================
// Main FSM
// ============================================================
always @(posedge clk or posedge reset) begin
    if (reset || ctrl_reset) begin
        state          <= S_IDLE;
        status_busy    <= 1'b0;
        status_done    <= 1'b0;
        cycle_count    <= 32'd0;
        sample_cnt     <= 10'd0;
        chirp_cnt      <= 7'd0;
        sram_base      <= 16'd0;
        timer          <= 32'd0;
        isram_addr     <= 16'd0;
        isram_rd       <= 1'b0;
        osram_addr     <= 16'd0;
        osram_wr       <= 1'b0;
        osram_wrdata   <= 32'd0;
        fft_sink_valid <= 1'b0;
        fft_sink_data  <= 32'd0;
        fft_sink_sop   <= 1'b0;
        fft_sink_eop   <= 1'b0;
        fft_source_ready <= 1'b1;
    end else begin

        // ---- Free running timer (counts while BUSY) ----
        if (status_busy) timer <= timer + 1;

        // ---- Default: deassert one-cycle signals ----
        isram_rd       <= 1'b0;
        osram_wr       <= 1'b0;
        fft_sink_valid <= 1'b0;
        fft_sink_sop   <= 1'b0;
        fft_sink_eop   <= 1'b0;

        case (state)

            // ------------------------------------------------
            S_IDLE: begin
                status_busy <= 1'b0;
                if (ctrl_start) begin
                    status_busy <= 1'b1;
                    status_done <= 1'b0;
                    chirp_cnt   <= 7'd0;
                    sram_base   <= 16'd0;
                    timer       <= 32'd0;
                    state       <= S_LOAD;
                end
            end

            // ------------------------------------------------
            // Read current sample from input SRAM
            S_LOAD: begin
                isram_addr <= sram_base + sample_cnt[15:0];
                isram_rd   <= 1'b1;
                state      <= S_LOAD_WAIT;
            end

            // ------------------------------------------------
            // Wait one cycle for SRAM read to complete
            S_LOAD_WAIT: begin
                state <= S_FEED_FFT;
            end

            // ------------------------------------------------
            // Push sample into FFT — wait if FFT not ready
            S_FEED_FFT: begin
                if (fft_sink_ready) begin
                    fft_sink_valid <= 1'b1;
                    fft_sink_data  <= isram_rddata;
                    fft_sink_sop   <= (sample_cnt == 10'd0);
                    fft_sink_eop   <= (sample_cnt == cfg_nr - 1);

                    if (sample_cnt == cfg_nr - 1) begin
                        // Last sample of this chirp — wait for FFT output
                        sample_cnt <= 10'd0;
                        state      <= S_WAIT_FFT;
                    end else begin
                        sample_cnt <= sample_cnt + 1;
                        state      <= S_LOAD;  // get next sample
                    end
                end
                // If FFT not ready, stay here and retry
            end

            // ------------------------------------------------
            // Consume FFT output bins, compute magnitude, write SRAM
            S_WAIT_FFT: begin
                fft_source_ready <= 1'b1;
                if (fft_source_valid) begin
                    state <= S_WRITE_MAG;
                end
            end

            // ------------------------------------------------
            S_WRITE_MAG: begin
                osram_addr   <= sram_base + sample_cnt[15:0];
                osram_wr     <= 1'b1;
                osram_wrdata <= mag_sq;

                if (sample_cnt == cfg_nr - 1) begin
                    // Done with this chirp's output
                    sample_cnt <= 10'd0;
                    state      <= S_NEXT_CHIRP;
                end else begin
                    sample_cnt <= sample_cnt + 1;
                    state      <= S_WAIT_FFT; // get next FFT bin
                end
            end

            // ------------------------------------------------
            S_NEXT_CHIRP: begin
                if (chirp_cnt == cfg_nd - 1) begin
                    // All chirps processed
                    state <= S_DONE;
                end else begin
                    chirp_cnt  <= chirp_cnt + 1;
                    sram_base  <= sram_base + cfg_nr;
                    state      <= S_LOAD;
                end
            end

            // ------------------------------------------------
            S_DONE: begin
                status_busy  <= 1'b0;
                status_done  <= 1'b1;
                cycle_count  <= timer;
                state        <= S_IDLE;
            end

            default: state <= S_IDLE;

        endcase
    end
end

endmodule
