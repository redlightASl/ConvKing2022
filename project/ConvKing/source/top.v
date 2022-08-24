`timescale 1ns/1ps
`include "Uart/fruit_defines.v"

module top(
           //system inout
           input sys_clk,
           input rst_n,

           //uart inout
           input uart_rx,
           output uart_tx,

           //key in
           input [1: 0] key_button,
           input weight_ctrl_button,

           //led out
           output [2: 0] led_out,
           output led_t,

           //SD card
           output sd_ncs,
           output sd_dclk,
           output sd_mosi,             //SD card controller data output
           input sd_miso,             //SD card controller data input

           //hdmi output
           output tmds_clk_p,
           output tmds_clk_n,
           output[2: 0] tmds_data_p,
           output[2: 0] tmds_data_n,

           //ov5640 inout
           inout cmos_scl,
           inout cmos_sda,
           input cmos_vsync,
           input cmos_href,
           input cmos_pclk,
           output cmos_xclk,
           input [7: 0] cmos_db,

           //ddr inout
           input pad_loop_in,
           input pad_loop_in_h,
           output pad_rstn_ch0,
           output pad_ddr_clk_w,
           output pad_ddr_clkn_w,
           output pad_csn_ch0,
           output [16 - 1: 0] pad_addr_ch0,
           inout [16 - 1: 0] pad_dq_ch0,
           inout [16 / 8 - 1: 0] pad_dqs_ch0,
           inout [16 / 8 - 1: 0] pad_dqsn_ch0,
           output [16 / 8 - 1: 0] pad_dm_rdqs_ch0,
           output pad_cke_ch0,
           output pad_odt_ch0,
           output pad_rasn_ch0,
           output pad_casn_ch0,
           output pad_wen_ch0,
           output [2: 0] pad_ba_ch0,
           output pad_loop_out,
           output pad_loop_out_h
           //    output pll_lock,
           //    output ddr_init_done,
           //    output ddrphy_rst_done,
       );

parameter MEM_DATA_BITS = 64; //external memory user interface data width
parameter ADDR_BITS = 24; //external memory user interface address width
parameter BUSRT_BITS = 10; //external memory user interface burst width
parameter BURST_SIZE = 64;

parameter CAM_OUTPUT_DATA_WIDTH = 16;
parameter HDMI_INOUT_DATA_WIDTH = 24; //HDMI raw input/output data width
parameter YOLO_INPUT_BITWIDTH = 16;
parameter YOLO_OUTPUT_BITWIDTH = 16;
// parameter YOLO_OUTPUT_BITWIDTH = 24;

wire ddr_init_done; //ddr init_done signal

//Timing Controller BUFG
wire video_clk5x;
wire video_clk;

wire sys_clk_g;
wire cmos_pclk_g;
wire cmos_xclk_w;
wire video_clk5x_w;
wire video_clk_w;
wire sd_card_clk;

//sysclk in
GTP_CLKBUFG sys_clkbufg(
                .CLKIN (sys_clk ),
                .CLKOUT (sys_clk_g ) // sys_clk in
            );
//pclk in
GTP_CLKBUFG cmos_pclkbufg(
                .CLKIN (cmos_pclk ),
                .CLKOUT (cmos_pclk_g ) // cmos_pclk in
            );
//System PLL
wire rst_n_eep;
video_pll video_pll_m0 (
              .clkin1(sys_clk_g),
              .pll_rst(1'b0),
              // .pll_rst(~rst_n),

              .clkout0(video_clk_w),
              .clkout1(video_clk5x_w),
              .clkout2(cmos_xclk_w ),
              .pll_lock(rst_n_eep)
          );
//xclk out
GTP_CLKBUFG cmos_xclkbufg(
                .CLKIN (cmos_xclk_w ),
                .CLKOUT (cmos_xclk ) // cmos_xclk out
            );
//videoclk 5x out
GTP_CLKBUFG video_clk5xbufg(
                .CLKIN (video_clk5x_w ),
                .CLKOUT (video_clk5x ) // video_clk5x out
            );
//videoclk out
GTP_CLKBUFG video_clkbufg(
                .CLKIN (video_clk_w ),
                .CLKOUT (video_clk ) // video_clk out
            );
//SD card controller clock
assign sd_card_clk = ~sys_clk_g;

//Key FSM
wire is_count_mode_o;
wire is_color_mode_o;
wire is_detect_mode_o;
key_top #(
            .DELAY_TIME ( 20'd1_000_000 )
        )
        u_key_top(
            //ports
            .sys_clk_i ( sys_clk_g ),
            .sys_rst_n_i ( rst_n ),
            .key_button_i ( key_button[1: 0] ),
            .led_indicate_o ( led_out[2: 0] ),
            .is_count_mode_o ( is_count_mode_o ),
            .is_color_mode_o ( is_color_mode_o ),
            .is_detect_mode_o ( is_detect_mode_o )
        );

//SD card
wire [3: 0] state_code;
wire mmc_write_req;
wire mmc_write_req_ack;
wire mmc_write_en;
wire [31: 0] mmc_write_data;

assign led_t = state_code[0];

sd_card_weight sd_card_weight_inst(
                   .clk (sd_card_clk ),
                   .rst (~rst_n ),

                   .key (weight_ctrl_button ),
                   .SD_nCS (sd_ncs ),
                   .SD_DCLK (sd_dclk ),
                   .SD_MOSI (sd_mosi ),
                   .SD_MISO (sd_miso ),

                   .state_code (state_code ),
                   .bmp_width (16'd1024 ), //image width

                   .write_req (mmc_write_req ),
                   .write_req_ack (mmc_write_req_ack ),
                   .write_en (mmc_write_en ),
                   .write_data (mmc_write_data )
               );

//OV5640 Controller
//I2C master controller
wire[9: 0] lut_index;
wire[31: 0] lut_data;
i2c_config i2c_config_m0 (
               .rst (~ddr_init_done),
               .clk (sys_clk_g),
               .clk_div_cnt (16'd500),
               .i2c_addr_2byte (1'b1),
               .lut_index (lut_index),
               .lut_dev_addr (lut_data[31: 24]),
               .lut_reg_addr (lut_data[23: 8]),
               .lut_reg_data (lut_data[7: 0]),
               .error (),
               .done (),
               .i2c_scl (cmos_scl),
               .i2c_sda (cmos_sda)
           );
//configure LUT
lut_ov5640_rgb565_1024_768 lut_ov5640_rgb565_1024_768_m0 (
                               .lut_index (lut_index),
                               .lut_data (lut_data)
                           );

//CMOS sensor 8bit data is converted to 16bit data
wire[15: 0] cmos_16bit_data;
wire cmos_16bit_wr;
cmos_8_16bit cmos_8_16bit_m0 (
                 .rst (~rst_n ),
                 .pclk (cmos_pclk_g ),
                 .pdata_i (cmos_db ),
                 .de_i (cmos_href ),
                 .pdata_o (cmos_16bit_data ),
                 .hblank (),
                 .de_o (cmos_16bit_wr )
             );

//CMOS sensor writes the request and generates the read and write address index
wire[1: 0] cmos_write_addr_index;
wire[1: 0] cmos_read_addr_index;
wire cmos_write_req;
wire cmos_write_req_ack;
//OV5640 data write gerneration
wire cmos_write_en;
wire[15: 0] cmos_write_data;
assign cmos_write_en = cmos_16bit_wr;
assign cmos_write_data = {cmos_16bit_data[4: 0], cmos_16bit_data[10: 5], cmos_16bit_data[15: 11]};
cmos_write_req_gen cmos_write_req_gen_m0 (
                       .rst (~rst_n ),
                       .pclk (cmos_pclk_g ),
                       .cmos_vsync (cmos_vsync ),
                       .write_req (cmos_write_req ),
                       .write_addr_index (cmos_write_addr_index ),
                       .read_addr_index (cmos_read_addr_index ),
                       .write_req_ack (cmos_write_req_ack )
                   );














//Video Output Timing Generator
//read data from ddr
//!output sync and vout to YOLO
wire read_req;
wire read_req_ack;
wire read_en;

wire hs;
wire vs;
wire de;

wire[CAM_OUTPUT_DATA_WIDTH - 1: 0] read_data;
wire [CAM_OUTPUT_DATA_WIDTH - 1: 0] vout_data;
video_timing_data #(
                      .DATA_WIDTH(CAM_OUTPUT_DATA_WIDTH) // Video data one clock data width
                  )
                  video_timing_data_m0
                  (
                      .video_clk (video_clk ),
                      .rst (~rst_n ),

                      .read_req (read_req ),
                      .read_req_ack (read_req_ack ),
                      .read_en (read_en ),
                      .read_data (read_data ),

                      .hs (hs ),
                      .vs (vs ),
                      .de (de ),
                      .vout_data (vout_data)
                  );

// //!auxiliary codes begin
// //raw RGB data analysis
// wire [7: 0] test_r;
// wire [7: 0] test_g;
// wire [7: 0] test_b;
// //?rgb565
// // assign test_r = {vout_data[15: 11], 3'd0};
// // assign test_g = {vout_data[10: 5], 2'd0};
// // assign test_b = {vout_data[4: 0], 3'd0};
// //?rgb888
// assign test_r = vout_data[23: 16];
// assign test_g = vout_data[15: 8];
// assign test_b = vout_data[7: 0];
// //HDMI Controller
// dvi_encoder dvi_encoder_m0 (
//                 //sys data
//                 .pixelclk (video_clk ),    // system clock
//                 .pixelclk5x (video_clk5x ),    // system clock x5
//                 .rstin (~rst_n ),    // reset

//                 //input data
//                 .blue_din (test_b),    // Blue data in
//                 .green_din (test_g),    // Green data in
//                 .red_din (test_r),    // Red data in

//                 //clk data
//                 .hsync (hs ),    // hsync data
//                 .vsync (vs ),    // vsync data
//                 .de (de ),    // data enable

//                 //output
//                 .tmds_clk_p (tmds_clk_p ),
//                 .tmds_clk_n (tmds_clk_n ),
//                 .tmds_data_p (tmds_data_p ),
//                 .tmds_data_n (tmds_data_n )
//             );
// //!auxiliary codes end

//raw RGB data analysis
wire [YOLO_OUTPUT_BITWIDTH - 1: 0] yolo_data_out;

//rgb565 to rgb888
// wire [7:0] cam_extend_r;
// wire [7:0] cam_extend_g;
// wire [7:0] cam_extend_b;
// assign cam_extend_r = {vout_data[15: 11], 3'd0};
// assign cam_extend_g = {vout_data[10: 5], 2'd0};
// assign cam_extend_b = {vout_data[4: 0], 3'd0};

yolo_top #(
             .YOLO_INPUT_BITWIDTH(YOLO_INPUT_BITWIDTH),
             .YOLO_OUTPUT_BITWIDTH(YOLO_OUTPUT_BITWIDTH)
         )
         yolo_inst (
             //system
             .sys_clk(video_clk),
             .sys_rst_n(rst_n),

             //input data
            //  .rgb_i({cam_extend_r, cam_extend_g, cam_extend_b}),
            .rgb_i(vout_data),
             //HDMI control sync timing trans
             .hsync_i (hs),
             .vsync_i (vs),
             .de_i (de),

             //output data
             .data_out(yolo_data_out),
             //output ctrl data
             .yolo_hsync(yolo_hsync),
             .yolo_vsync(yolo_vsync),
             .yolo_de (yolo_de)
         );

//KNN Color Detection
wire [2: 0] knn_color;
color_det_top #(
                  .KNN_INPUT_BITWIDTH(YOLO_INPUT_BITWIDTH)
              )
              color_det_top_inst (
                  //system
                  .sys_clk(video_clk),
                  .sys_rst_n(rst_n),

                  //input data
                  .rgb_i({cam_extend_r, cam_extend_g, cam_extend_b}),
                  //HDMI control sync timing trans
                  .hsync_i (hs),
                  .vsync_i (vs),
                  .de_i (de),

                  //output data
                  .knn_color(knn_color)
              );

//YOLO output data analysis
wire [7: 0] hdmi_r;
wire [7: 0] hdmi_g;
wire [7: 0] hdmi_b;
//?rgb565
assign hdmi_r = {yolo_data_out[15: 11], 3'd0};
assign hdmi_g = {yolo_data_out[10: 5], 2'd0};
assign hdmi_b = {yolo_data_out[4: 0], 3'd0};
//?rgb888
// assign hdmi_r = yolo_data_out[23:16];
// assign hdmi_g = yolo_data_out[15:8];
// assign hdmi_b = yolo_data_out[7:0];

//HDMI Controller
dvi_encoder dvi_encoder_m0 (
                //sys data
                .pixelclk (video_clk ),  // system clock
                .pixelclk5x (video_clk5x ),  // system clock x5
                .rstin (~rst_n ),  // reset

                //input data
                .blue_din (hdmi_b),  // Blue data in
                .green_din (hdmi_g),  // Green data in
                .red_din (hdmi_r),  // Red data in

                //clk data
                .hsync (yolo_hsync ),  // hsync data
                .vsync (yolo_vsync ),  // vsync data
                .de (yolo_de ),  // data enable

                //output
                .tmds_clk_p (tmds_clk_p ),
                .tmds_clk_n (tmds_clk_n ),
                .tmds_data_p (tmds_data_p ),
                .tmds_data_n (tmds_data_n )
            );

//UART
uart_decode #(
                .CLK_FRE ( 50 ),
                .BAUD_RATE ( 115200 )
            )
            u_uart_decode (
                .sys_clk ( sys_clk_g ),
                .rst_n ( rst_n ),

                .is_count_mode_i ( is_count_mode_o ),
                .is_color_mode_i ( is_color_mode_o ),
                .is_detect_mode_i ( is_detect_mode_o ),

                .fruit_number ( 8'd1 ),
                .color ( knn_color ),
                .subject ( `SUB_GRAPE ),

                .uart_rx ( uart_rx ),
                .uart_tx ( uart_tx )
            );



// uart_top #(
//     .CLK_FRE   ( 50     ),
//     .BAUD_RATE ( 115200 ))
//  u_uart_top (
//     .sys_clk                 ( sys_clk_g               ),
//     .uart_fifo_write_clk     ( sd_card_clk   ),
//     .rst_n                   ( rst_n                 ),

//     .write_req               ( mmc_write_req             ),
//     .write_en                ( mmc_write_en              ),
//     .write_data              ( mmc_write_data            ),
//     .uart_rx                 ( uart_rx               ),

//     .write_req_ack           ( mmc_write_req_ack         ),
//     .uart_tx                 ( uart_tx               )
// );































//AXI System Interconnection
wire axi_clk;
wire wr_burst_data_req;
wire wr_burst_finish;
wire rd_burst_finish;
wire rd_burst_req;
wire wr_burst_req;
wire[BUSRT_BITS - 1: 0] rd_burst_len;
wire[BUSRT_BITS - 1: 0] wr_burst_len;
wire[ADDR_BITS - 1: 0] rd_burst_addr;
wire[ADDR_BITS - 1: 0] wr_burst_addr;
wire rd_burst_data_valid;
wire[MEM_DATA_BITS - 1 : 0] rd_burst_data;
wire[MEM_DATA_BITS - 1 : 0] wr_burst_data;

//Frame R/W Controller
frame_read_write # (
                     .MEM_DATA_BITS (MEM_DATA_BITS ),
                     .READ_DATA_BITS (YOLO_INPUT_BITWIDTH ),
                     .WRITE_DATA_BITS (YOLO_INPUT_BITWIDTH ),
                     .ADDR_BITS (ADDR_BITS ),
                     .BUSRT_BITS (BUSRT_BITS ),
                     .BURST_SIZE (BURST_SIZE )
                 ) frame_read_write_m0 (
                     .rst (~rst_n),
                     .mem_clk (axi_clk ),
                     .read_clk (video_clk ),
                     .write_clk (cmos_pclk_g ),


                     //read data by video_timing_data_m0
                     .rd_burst_req (rd_burst_req ),
                     .rd_burst_len (rd_burst_len ),
                     .rd_burst_addr (rd_burst_addr ),
                     .rd_burst_data_valid (rd_burst_data_valid ),
                     .rd_burst_data (rd_burst_data ),
                     .rd_burst_finish (rd_burst_finish ),
                     .read_req (read_req ),
                     .read_req_ack (read_req_ack ),
                     .read_finish ( ),
                     //!The first frame address is 0
                     //!The second frame address is 24'd2073600 ,large enough address space for one frame of video
                     .read_addr_0 ({ADDR_BITS{1'b0}} ),
                     //  .read_addr_1 (24'd2073600 ), //+2073600
                     //  .read_addr_2 (24'd4147200 ), //+2073600
                     //  .read_addr_3 (24'd6220800 ), //+2073600
                     .read_addr_1 (24'd196609 ),
                     .read_addr_2 (24'd393217 ),
                     .read_addr_3 (24'd589826 ),
                     .read_addr_index (cmos_read_addr_index),
                     .read_len (24'd196608),  //frame size
                     .read_en (read_en ),
                     .read_data (read_data ),

                     //write data by OV2560 cmos_write_req_gen_m0
                     .wr_burst_req (wr_burst_req ),
                     .wr_burst_len (wr_burst_len ),
                     .wr_burst_addr (wr_burst_addr ),
                     .wr_burst_data_req (wr_burst_data_req ),
                     .wr_burst_data (wr_burst_data ),
                     .wr_burst_finish (wr_burst_finish ),

                     .write_req (cmos_write_req ),
                     .write_req_ack (cmos_write_req_ack ),
                     .write_finish ( ),
                     .write_addr_0 ({ADDR_BITS{1'b0}} ),
                     //  .write_addr_1 (24'd2073600 ),
                     //  .write_addr_2 (24'd4147200 ),
                     //  .write_addr_3 (24'd6220800 ),
                     .write_addr_1 (24'd196609 ),
                     .write_addr_2 (24'd393217 ),
                     .write_addr_3 (24'd589826 ),
                     .write_addr_index (cmos_write_addr_index ),
                     .write_len (24'd196608 ),  //frame size 1024 * 768 * 16 / 64
                     .write_en (cmos_write_en ),
                     .write_data (cmos_write_data )

                     //  .read_addr_0 (24'd0 ),
                     //  .read_addr_1 (24'd0 ),
                     //  .read_addr_2 (24'd0 ),
                     //  .read_addr_3 (24'd0 ),
                     //  .read_addr_index (2'd0 ), //use only read_addr_0
                     //  .read_len (24'd393216 ), //frame size 1024 * 768 * 32 / 64
                     //  .write_clk (sd_card_clk ),
                     //  .write_req (mmc_write_req ),
                     //  .write_req_ack (mmc_write_req_ack ),
                     //  .write_addr_0 (24'd0 ),
                     //  .write_addr_1 (24'd0 ),
                     //  .write_addr_2 (24'd0 ),
                     //  .write_addr_3 (24'd0 ),
                     //  .write_addr_index (2'd0 ),   //use only write_addr_0
                     //  .write_len (24'd393216 ),   //frame size
                     //  .write_en (mmc_write_en ),
                     //  .write_data (mmc_write_data )
                 );










//DDR PHY Controller
localparam MEM_ROW_ADDRESS = 14;
localparam MEM_COLUMN_ADDRESS = 10;
localparam MEM_BANK_ADDRESS = 3;
localparam CTRL_ADDR_WIDTH = MEM_ROW_ADDRESS + MEM_COLUMN_ADDRESS + MEM_BANK_ADDRESS;

//AXI Bus Interconnection
// Master Write Address
wire [3: 0] s00_axi_awid;
wire [63: 0] s00_axi_awaddr;
wire [7: 0] s00_axi_awlen;    // burst length: 0-255
wire [2: 0] s00_axi_awsize;   // burst size: fixed 2'b011
wire [1: 0] s00_axi_awburst;  // burst type: fixed 2'b01(incremental burst)
wire s00_axi_awlock;   // lock: fixed 2'b00
wire [3: 0] s00_axi_awcache;  // cache: fiex 2'b0011
wire [2: 0] s00_axi_awprot;   // protect: fixed 2'b000
wire [3: 0] s00_axi_awqos;    // qos: fixed 2'b0000
wire [0: 0] s00_axi_awuser;   // user: fixed 32'd0
wire s00_axi_awvalid;
wire s00_axi_awready;
// master write data
wire [63: 0] s00_axi_wdata;
wire [7: 0] s00_axi_wstrb;
wire s00_axi_wlast;
wire [0: 0] s00_axi_wuser;
wire s00_axi_wvalid;
wire s00_axi_wready;
// master write response
wire [3: 0] s00_axi_bid;
wire [1: 0] s00_axi_bresp;
wire [0: 0] s00_axi_buser;
wire s00_axi_bvalid;
wire s00_axi_bready;
// master read address
wire [3: 0] s00_axi_arid;
wire [63: 0] s00_axi_araddr;
wire [7: 0] s00_axi_arlen;
wire [2: 0] s00_axi_arsize;
wire [1: 0] s00_axi_arburst;
wire [1: 0] s00_axi_arlock;
wire [3: 0] s00_axi_arcache;
wire [2: 0] s00_axi_arprot;
wire [3: 0] s00_axi_arqos;
wire [0: 0] s00_axi_aruser;
wire s00_axi_arvalid;
wire s00_axi_arready;
// master read data
wire [3: 0] s00_axi_rid;
wire [63: 0] s00_axi_rdata;
wire [1: 0] s00_axi_rresp;
wire s00_axi_rlast;
wire [0: 0] s00_axi_ruser;
wire s00_axi_rvalid;
wire s00_axi_rready;

wire s00_axi_awurgent;
wire s00_axi_awpoison;
wire s00_axi_arpoison;
wire s00_axi_arurgent;

wire pll_lock;
wire ddrphy_rst_done;

assign s00_axi_awurgent = 1'b0;
assign s00_axi_awpoison = 1'b0;
assign s00_axi_arpoison = 1'b0;
assign s00_axi_arurgent = 1'b0;

ddr3_core u_ipsl_hmic_h_top (
              .pll_refclk_in (sys_clk_g ),      //system clock
              .top_rst_n (rst_n ),      //system reset

              //aclk
              .pll_aclk_0 ( ),
              .pll_aclk_1 (axi_clk ),
              .pll_aclk_2 ( ),

              .pll_lock (pll_lock ),
              .ddrphy_rst_done (ddrphy_rst_done),
              .ddrc_init_done (ddr_init_done ),
              .ddrc_rst (1'b0),

              .areset_1 (1'b0),
              .aclk_1 (axi_clk),
              .awid_1 (s00_axi_awid),
              .awaddr_1 (s00_axi_awaddr),
              .awlen_1 (s00_axi_awlen),
              .awsize_1 (s00_axi_awsize),
              .awburst_1 (s00_axi_awburst),
              .awlock_1 (s00_axi_awlock),
              .awvalid_1 (s00_axi_awvalid),
              .awready_1 (s00_axi_awready),
              .awurgent_1 (s00_axi_awurgent),
              .awpoison_1 (s00_axi_awpoison),

              .wdata_1 (s00_axi_wdata),
              .wstrb_1 (s00_axi_wstrb),
              .wlast_1 (s00_axi_wlast),
              .wvalid_1 (s00_axi_wvalid),
              .wready_1 (s00_axi_wready),
              .bid_1 (s00_axi_bid),
              .bresp_1 (s00_axi_bresp),
              .bvalid_1 (s00_axi_bvalid),
              .bready_1 (s00_axi_bready),
              .arid_1 (s00_axi_arid ),
              .araddr_1 (s00_axi_araddr ),
              .arlen_1 (s00_axi_arlen ),
              .arsize_1 (s00_axi_arsize ),
              .arburst_1 (s00_axi_arburst ),
              .arlock_1 (s00_axi_arlock ),
              .arvalid_1 (s00_axi_arvalid ),
              .arready_1 (s00_axi_arready ),
              .arpoison_1 (s00_axi_arpoison),

              .rid_1 (s00_axi_rid ),
              .rdata_1 (s00_axi_rdata ),
              .rresp_1 (s00_axi_rresp ),
              .rlast_1 (s00_axi_rlast ),
              .rvalid_1 (s00_axi_rvalid ),
              .rready_1 (s00_axi_rready ),
              .arurgent_1 (s00_axi_arurgent),

              .csysreq_1 (1'b1),
              .csysack_1 (),
              .cactive_1 (),

              .csysreq_ddrc (1'b1),
              .csysack_ddrc (),
              .cactive_ddrc (),

              //outpins
              .pad_loop_in (pad_loop_in),
              .pad_loop_in_h (pad_loop_in_h),
              .pad_rstn_ch0 (pad_rstn_ch0),
              .pad_ddr_clk_w (pad_ddr_clk_w),
              .pad_ddr_clkn_w (pad_ddr_clkn_w),
              .pad_csn_ch0 (pad_csn_ch0),
              .pad_addr_ch0 (pad_addr_ch0),
              .pad_dq_ch0 (pad_dq_ch0),
              .pad_dqs_ch0 (pad_dqs_ch0),
              .pad_dqsn_ch0 (pad_dqsn_ch0),
              .pad_dm_rdqs_ch0 (pad_dm_rdqs_ch0),
              .pad_cke_ch0 (pad_cke_ch0),
              .pad_odt_ch0 (pad_odt_ch0),
              .pad_rasn_ch0 (pad_rasn_ch0),
              .pad_casn_ch0 (pad_casn_ch0),
              .pad_wen_ch0 (pad_wen_ch0),
              .pad_ba_ch0 (pad_ba_ch0),
              .pad_loop_out (pad_loop_out),
              .pad_loop_out_h (pad_loop_out_h)
          );
//AXI Controller
aq_axi_master u_aq_axi_master
              (
                  .ARESETN (rst_n),
                  .ACLK (axi_clk ),
                  .M_AXI_AWID (s00_axi_awid ),
                  .M_AXI_AWADDR (s00_axi_awaddr ),
                  .M_AXI_AWLEN (s00_axi_awlen ),
                  .M_AXI_AWSIZE (s00_axi_awsize ),
                  .M_AXI_AWBURST (s00_axi_awburst ),
                  .M_AXI_AWLOCK (s00_axi_awlock ),
                  .M_AXI_AWCACHE (s00_axi_awcache ),
                  .M_AXI_AWPROT (s00_axi_awprot ),
                  .M_AXI_AWQOS (s00_axi_awqos ),
                  .M_AXI_AWUSER (s00_axi_awuser ),
                  .M_AXI_AWVALID (s00_axi_awvalid ),
                  .M_AXI_AWREADY (s00_axi_awready ),
                  .M_AXI_WDATA (s00_axi_wdata ),
                  .M_AXI_WSTRB (s00_axi_wstrb ),
                  .M_AXI_WLAST (s00_axi_wlast ),
                  .M_AXI_WUSER (s00_axi_wuser ),
                  .M_AXI_WVALID (s00_axi_wvalid ),
                  .M_AXI_WREADY (s00_axi_wready ),
                  .M_AXI_BID (s00_axi_bid ),
                  .M_AXI_BRESP (s00_axi_bresp ),
                  .M_AXI_BUSER (s00_axi_buser ),
                  .M_AXI_BVALID (s00_axi_bvalid ),
                  .M_AXI_BREADY (s00_axi_bready ),
                  .M_AXI_ARID (s00_axi_arid ),
                  .M_AXI_ARADDR (s00_axi_araddr ),
                  .M_AXI_ARLEN (s00_axi_arlen ),
                  .M_AXI_ARSIZE (s00_axi_arsize ),
                  .M_AXI_ARBURST (s00_axi_arburst ),
                  .M_AXI_ARLOCK (s00_axi_arlock ),
                  .M_AXI_ARCACHE (s00_axi_arcache ),
                  .M_AXI_ARPROT (s00_axi_arprot ),
                  .M_AXI_ARQOS (s00_axi_arqos ),
                  .M_AXI_ARUSER (s00_axi_aruser ),
                  .M_AXI_ARVALID (s00_axi_arvalid ),
                  .M_AXI_ARREADY (s00_axi_arready ),
                  .M_AXI_RID (s00_axi_rid ),
                  .M_AXI_RDATA (s00_axi_rdata ),
                  .M_AXI_RRESP (s00_axi_rresp ),
                  .M_AXI_RLAST (s00_axi_rlast ),
                  .M_AXI_RUSER (s00_axi_ruser ),
                  .M_AXI_RVALID (s00_axi_rvalid ),
                  .M_AXI_RREADY (s00_axi_rready ),
                  .MASTER_RST (1'b0 ),
                  .WR_START (wr_burst_req ),
                  .WR_ADRS ({wr_burst_addr, 3'd0} ),
                  .WR_LEN ({wr_burst_len, 3'd0} ),
                  .WR_READY ( ),
                  .WR_FIFO_RE (wr_burst_data_req ),
                  .WR_FIFO_EMPTY (1'b0 ),
                  .WR_FIFO_AEMPTY (1'b0 ),
                  .WR_FIFO_DATA (wr_burst_data ),
                  .WR_DONE (wr_burst_finish ),
                  .RD_START (rd_burst_req ),
                  .RD_ADRS ({rd_burst_addr, 3'd0} ),
                  .RD_LEN ({rd_burst_len, 3'd0} ),
                  .RD_READY ( ),
                  .RD_FIFO_WE (rd_burst_data_valid ),
                  .RD_FIFO_FULL (1'b0 ),
                  .RD_FIFO_AFULL (1'b0 ),
                  .RD_FIFO_DATA (rd_burst_data ),
                  .RD_DONE (rd_burst_finish ),
                  .DEBUG ( )
              );













//-------------debugger-------------
/*
wire hub_tdi;
wire hub_tdo;
wire [4: 0] debug_id;
wire debug_drck;
wire debug_capt;
wire debug_shift;
wire conf_sel;

DebugCore InsertedDebugger (
              .clk(sys_clk), 
              .resetn_i(rst_n),

              .hub_tdi(hub_tdi),         
              .hub_tdo(hub_tdo),         

              .id_i(debug_id), // input [4:0]
              .drck_in(debug_drck),      
              .capt_i(debug_capt),       
              .shift_i(shift_i),         
              .conf_sel(conf_sel),       

              .trig0_i()       
          );

JtagCore InsertedJtag (
             .resetn_i(rst_n),       

             .hub_tdi(hub_tdo),      
             .hub_tdo(hub_tdi), // input [14:0]

             .id_o(debug_id), // output [4:0]
             .drck_o(debug_drck),    
             .capt_o(debug_capt),    
             .shift_o(debug_shift),  
             .conf_sel(conf_sel) // output [14:0]
         );
*/
endmodule
