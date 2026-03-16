// ============================================================
// avalon_ctrl_slave.v
// Avalon MM Slave — Radar Accelerator Control Registers
// COE838 — FMCW Radar Range-Doppler SoC
//
// Register Map (word-addressed on lightweight bridge):
//   0x00  CTRL    [0]=START(pulse) [1]=RESET
//   0x01  STATUS  [0]=BUSY        [1]=DONE
//   0x02  NR      Range FFT length (default 256)
//   0x03  ND      Number of chirps (default 64)
//   0x04  CYCLES  Accelerator cycle count (read-only)
//
// NOTE: avs_* prefix is recognized by QSys as Avalon MM Slave
// NOTE: ctrl_* and status_* become conduit exports in QSys
// ============================================================

module avalon_ctrl_slave (
    // Clock and reset (QSys connects these automatically)
    input  wire        clk,
    input  wire        reset,          // active-high from QSys

    // ---- Avalon MM Slave Interface (auto-detected by QSys) ----
    input  wire [2:0]  avs_address,    // which register (0-4)
    input  wire        avs_write,      // HPS is writing
    input  wire        avs_read,       // HPS is reading
    input  wire [31:0] avs_writedata,  // data from HPS
    output reg  [31:0] avs_readdata,   // data back to HPS

    // ---- Conduit: Control outputs → Accelerator FSM ----
    output reg         ctrl_start,     // pulsed 1 cycle on START
    output reg         ctrl_reset,     // held high while reset bit set
    output reg  [9:0]  cfg_nr,         // range FFT length
    output reg  [6:0]  cfg_nd,         // number of chirps

    // ---- Conduit: Status inputs ← Accelerator FSM ----
    input  wire        status_busy,
    input  wire        status_done,
    input  wire [31:0] cycle_count
);

// ---- Register address constants ----
localparam CTRL_ADDR   = 3'd0;
localparam STATUS_ADDR = 3'd1;
localparam NR_ADDR     = 3'd2;
localparam ND_ADDR     = 3'd3;
localparam CYCLES_ADDR = 3'd4;

// ============================================================
// WRITE LOGIC
// HPS writes to CTRL, NR, ND registers
// START is a single-cycle pulse (auto-clears next cycle)
// ============================================================
always @(posedge clk or posedge reset) begin
    if (reset) begin
        ctrl_start <= 1'b0;
        ctrl_reset <= 1'b0;
        cfg_nr     <= 10'd256;   // default: 256 samples per chirp
        cfg_nd     <= 7'd64;     // default: 64 chirps per frame
    end else begin
        // Auto-clear START after one cycle (it's a pulse not a level)
        ctrl_start <= 1'b0;

        if (avs_write) begin
            case (avs_address)
                CTRL_ADDR: begin
                    ctrl_start <= avs_writedata[0];  // bit 0 = START
                    ctrl_reset <= avs_writedata[1];  // bit 1 = RESET
                end
                NR_ADDR: cfg_nr <= avs_writedata[9:0];
                ND_ADDR: cfg_nd <= avs_writedata[6:0];
                // STATUS and CYCLES are read-only, ignore writes
                default: ;
            endcase
        end
    end
end

// ============================================================
// READ LOGIC
// HPS reads STATUS, NR, ND, CYCLES
// ============================================================
always @(*) begin
    case (avs_address)
        CTRL_ADDR:   avs_readdata = {30'b0, ctrl_reset, 1'b0}; // START always reads 0
        STATUS_ADDR: avs_readdata = {30'b0, status_done, status_busy};
        NR_ADDR:     avs_readdata = {22'b0, cfg_nr};
        ND_ADDR:     avs_readdata = {25'b0, cfg_nd};
        CYCLES_ADDR: avs_readdata = cycle_count;
        default:     avs_readdata = 32'hDEADBEEF; // debug sentinel
    endcase
end

endmodule
