// ============================================================
// radar_top.v
// Top-Level Entity — FMCW Radar Range-Doppler SoC
// COE838 — Systems-on-Chip Design
//
// Structure (mirrors Lab 4 appendix pattern):
//   1. Declares all DE1-SoC board pins (DDR3, Ethernet, USB etc)
//   2. Instantiates QSys-generated soc_system
//   3. Instantiates accelerator_fsm
//   4. Wires QSys conduit exports → accelerator_fsm ports
//
// ---- IMPORTANT AFTER QSYS GENERATION ----
// After you generate the soc_system in QSys, open:
//   soc_system/synthesis/soc_system.v  (or .qsys instantiation template)
// and verify the conduit port names below match exactly.
// QSys appends _0_ and the conduit name you chose, e.g.:
//   ctrl_slave_0_conduit_ctrl_start
// Update the port map below to match your actual generated names.
// ============================================================

module radar_top (
    // ---- 50MHz clock ----
    input  wire        CLOCK_50,

    // ---- HPS DDR3 Memory (required for HPS to boot) ----
    output wire [14:0] HPS_DDR3_ADDR,
    output wire [2:0]  HPS_DDR3_BA,
    output wire        HPS_DDR3_CAS_N,
    output wire        HPS_DDR3_CKE,
    output wire        HPS_DDR3_CK_N,
    output wire        HPS_DDR3_CK_P,
    output wire        HPS_DDR3_CS_N,
    output wire [3:0]  HPS_DDR3_DM,
    inout  wire [31:0] HPS_DDR3_DQ,
    inout  wire [3:0]  HPS_DDR3_DQS_N,
    inout  wire [3:0]  HPS_DDR3_DQS_P,
    output wire        HPS_DDR3_ODT,
    output wire        HPS_DDR3_RAS_N,
    output wire        HPS_DDR3_RESET_N,
    input  wire        HPS_DDR3_RZQ,
    output wire        HPS_DDR3_WE_N,

    // ---- HPS Ethernet ----
    output wire        HPS_ENET_GTX_CLK,
    inout  wire        HPS_ENET_MDIO,
    output wire        HPS_ENET_MDC,
    input  wire        HPS_ENET_RX_CLK,
    input  wire [3:0]  HPS_ENET_RX_DATA,
    input  wire        HPS_ENET_RX_DV,
    output wire [3:0]  HPS_ENET_TX_DATA,
    output wire        HPS_ENET_TX_EN,

    // ---- HPS SD Card ----
    inout  wire        HPS_SD_CMD,
    inout  wire [3:0]  HPS_SD_DATA,
    output wire        HPS_SD_CLK,

    // ---- HPS USB ----
    input  wire        HPS_USB_CLKOUT,
    inout  wire [7:0]  HPS_USB_DATA,
    input  wire        HPS_USB_DIR,
    input  wire        HPS_USB_NXT,
    output wire        HPS_USB_STP
);

// ============================================================
// Internal signals — QSys conduit wires
// These are the signals that cross the QSys boundary
// into your accelerator hardware
// ============================================================

// HPS reset output (we don't use it but must wire it)
wire hps_fpga_reset_n;

// Control slave → accelerator FSM
wire        conduit_ctrl_start;
wire        conduit_ctrl_reset;
wire [9:0]  conduit_cfg_nr;
wire [6:0]  conduit_cfg_nd;

// Accelerator FSM → control slave
wire        conduit_status_busy;
wire        conduit_status_done;
wire [31:0] conduit_cycle_count;

// Accelerator FSM → input SRAM (read port)
wire [15:0] conduit_isram_addr;
wire        conduit_isram_rd;
wire [31:0] conduit_isram_rddata;

// Accelerator FSM → output SRAM (write port)
wire [15:0] conduit_osram_addr;
wire        conduit_osram_wr;
wire [31:0] conduit_osram_wrdata;

// Accelerator FSM ↔ FFT wrapper
wire        conduit_fft_sink_valid;
wire [31:0] conduit_fft_sink_data;
wire        conduit_fft_sink_sop;
wire        conduit_fft_sink_eop;
wire        conduit_fft_sink_ready;
wire        conduit_fft_source_valid;
wire [31:0] conduit_fft_source_data;
wire        conduit_fft_source_sop;
wire        conduit_fft_source_eop;
wire        conduit_fft_source_ready;

// ============================================================
// QSys-generated SoC system instantiation
//
// After QSys generates HDL, go to:
//   Generate → Show Instantiation Template
// and replace the port names below with the exact generated names
// ============================================================
soc_system u0 (
    // Clock and reset
    .clk_clk                            (CLOCK_50),
    .reset_reset_n                      (1'b1),
    .hps_0_h2f_reset_reset_n            (hps_fpga_reset_n),

    // HPS DDR3 memory
    .memory_mem_a                       (HPS_DDR3_ADDR),
    .memory_mem_ba                      (HPS_DDR3_BA),
    .memory_mem_ck                      (HPS_DDR3_CK_P),
    .memory_mem_ck_n                    (HPS_DDR3_CK_N),
    .memory_mem_cke                     (HPS_DDR3_CKE),
    .memory_mem_cs_n                    (HPS_DDR3_CS_N),
    .memory_mem_ras_n                   (HPS_DDR3_RAS_N),
    .memory_mem_cas_n                   (HPS_DDR3_CAS_N),
    .memory_mem_we_n                    (HPS_DDR3_WE_N),
    .memory_mem_reset_n                 (HPS_DDR3_RESET_N),
    .memory_mem_dq                      (HPS_DDR3_DQ),
    .memory_mem_dqs                     (HPS_DDR3_DQS_P),
    .memory_mem_dqs_n                   (HPS_DDR3_DQS_N),
    .memory_mem_odt                     (HPS_DDR3_ODT),
    .memory_mem_dm                      (HPS_DDR3_DM),
    .memory_oct_rzqin                   (HPS_DDR3_RZQ),

    // HPS Ethernet
    .hps_io_hps_io_emac1_inst_TX_CLK    (HPS_ENET_GTX_CLK),
    .hps_io_hps_io_emac1_inst_TXD0      (HPS_ENET_TX_DATA[0]),
    .hps_io_hps_io_emac1_inst_TXD1      (HPS_ENET_TX_DATA[1]),
    .hps_io_hps_io_emac1_inst_TXD2      (HPS_ENET_TX_DATA[2]),
    .hps_io_hps_io_emac1_inst_TXD3      (HPS_ENET_TX_DATA[3]),
    .hps_io_hps_io_emac1_inst_RXD0      (HPS_ENET_RX_DATA[0]),
    .hps_io_hps_io_emac1_inst_MDIO      (HPS_ENET_MDIO),
    .hps_io_hps_io_emac1_inst_MDC       (HPS_ENET_MDC),
    .hps_io_hps_io_emac1_inst_RX_CTL    (HPS_ENET_RX_DV),
    .hps_io_hps_io_emac1_inst_TX_CTL    (HPS_ENET_TX_EN),
    .hps_io_hps_io_emac1_inst_RX_CLK    (HPS_ENET_RX_CLK),
    .hps_io_hps_io_emac1_inst_RXD1      (HPS_ENET_RX_DATA[1]),
    .hps_io_hps_io_emac1_inst_RXD2      (HPS_ENET_RX_DATA[2]),
    .hps_io_hps_io_emac1_inst_RXD3      (HPS_ENET_RX_DATA[3]),

    // HPS SD Card
    .hps_io_hps_io_sdio_inst_CMD        (HPS_SD_CMD),
    .hps_io_hps_io_sdio_inst_D0         (HPS_SD_DATA[0]),
    .hps_io_hps_io_sdio_inst_D1         (HPS_SD_DATA[1]),
    .hps_io_hps_io_sdio_inst_CLK        (HPS_SD_CLK),
    .hps_io_hps_io_sdio_inst_D2         (HPS_SD_DATA[2]),
    .hps_io_hps_io_sdio_inst_D3         (HPS_SD_DATA[3]),

    // HPS USB
    .hps_io_hps_io_usb1_inst_D0         (HPS_USB_DATA[0]),
    .hps_io_hps_io_usb1_inst_D1         (HPS_USB_DATA[1]),
    .hps_io_hps_io_usb1_inst_D2         (HPS_USB_DATA[2]),
    .hps_io_hps_io_usb1_inst_D3         (HPS_USB_DATA[3]),
    .hps_io_hps_io_usb1_inst_D4         (HPS_USB_DATA[4]),
    .hps_io_hps_io_usb1_inst_D5         (HPS_USB_DATA[5]),
    .hps_io_hps_io_usb1_inst_D6         (HPS_USB_DATA[6]),
    .hps_io_hps_io_usb1_inst_D7         (HPS_USB_DATA[7]),
    .hps_io_hps_io_usb1_inst_CLK        (HPS_USB_CLKOUT),
    .hps_io_hps_io_usb1_inst_STP        (HPS_USB_STP),
    .hps_io_hps_io_usb1_inst_DIR        (HPS_USB_DIR),
    .hps_io_hps_io_usb1_inst_NXT        (HPS_USB_NXT),

    // ---- Exported conduits — ctrl slave ----
    // These names come from QSys after you export the conduit.
    // Pattern: <component_name>_<conduit_name>_<signal_type>
    .ctrl_slave_0_conduit_ctrl_start    (conduit_ctrl_start),
    .ctrl_slave_0_conduit_ctrl_reset    (conduit_ctrl_reset),
    .ctrl_slave_0_conduit_cfg_nr        (conduit_cfg_nr),
    .ctrl_slave_0_conduit_cfg_nd        (conduit_cfg_nd),
    .ctrl_slave_0_conduit_status_busy   (conduit_status_busy),
    .ctrl_slave_0_conduit_status_done   (conduit_status_done),
    .ctrl_slave_0_conduit_cycle_count   (conduit_cycle_count),

    // ---- Exported conduits — input SRAM ----
    .input_sram_0_s2_address            (conduit_isram_addr),
    .input_sram_0_s2_read               (conduit_isram_rd),
    .input_sram_0_s2_readdata           (conduit_isram_rddata),
    .input_sram_0_s2_write              (1'b0),
    .input_sram_0_s2_writedata          (32'b0),

    // ---- Exported conduits — output SRAM ----
    .output_sram_0_s2_address           (conduit_osram_addr),
    .output_sram_0_s2_read              (1'b0),
    .output_sram_0_s2_readdata          (),    // not used
    .output_sram_0_s2_write             (conduit_osram_wr),
    .output_sram_0_s2_writedata         (conduit_osram_wrdata)
);

// ============================================================
// Accelerator FSM instantiation
// Receives conduit wires from QSys, drives SRAM and FFT
// ============================================================
accelerator_fsm accel_inst (
    .clk              (CLOCK_50),
    .reset            (~hps_fpga_reset_n),

    // Control slave interface
    .ctrl_start       (conduit_ctrl_start),
    .ctrl_reset       (conduit_ctrl_reset),
    .cfg_nr           (conduit_cfg_nr),
    .cfg_nd           (conduit_cfg_nd),
    .status_busy      (conduit_status_busy),
    .status_done      (conduit_status_done),
    .cycle_count      (conduit_cycle_count),

    // Input SRAM (read port)
    .isram_addr       (conduit_isram_addr),
    .isram_rd         (conduit_isram_rd),
    .isram_rddata     (conduit_isram_rddata),

    // Output SRAM (write port)
    .osram_addr       (conduit_osram_addr),
    .osram_wr         (conduit_osram_wr),
    .osram_wrdata     (conduit_osram_wrdata),

    // FFT interface
    .fft_sink_valid   (conduit_fft_sink_valid),
    .fft_sink_data    (conduit_fft_sink_data),
    .fft_sink_sop     (conduit_fft_sink_sop),
    .fft_sink_eop     (conduit_fft_sink_eop),
    .fft_sink_ready   (conduit_fft_sink_ready),
    .fft_source_valid (conduit_fft_source_valid),
    .fft_source_data  (conduit_fft_source_data),
    .fft_source_sop   (conduit_fft_source_sop),
    .fft_source_eop   (conduit_fft_source_eop),
    .fft_source_ready (conduit_fft_source_ready),
    .fft_inverse      ()
);

// ============================================================
// FFT Wrapper instantiation
// ============================================================
fft_wrapper fft_wrap_inst (
    .clk              (CLOCK_50),
    .reset            (~hps_fpga_reset_n),

    .sink_valid       (conduit_fft_sink_valid),
    .sink_data        (conduit_fft_sink_data),
    .sink_sop         (conduit_fft_sink_sop),
    .sink_eop         (conduit_fft_sink_eop),
    .sink_ready       (conduit_fft_sink_ready),

    .source_valid     (conduit_fft_source_valid),
    .source_data      (conduit_fft_source_data),
    .source_sop       (conduit_fft_source_sop),
    .source_eop       (conduit_fft_source_eop),
    .source_ready     (conduit_fft_source_ready),

    .inverse          (1'b0)
);

endmodule
