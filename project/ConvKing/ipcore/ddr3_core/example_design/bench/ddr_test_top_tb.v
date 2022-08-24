// Created by IP Generator (Version 2020.3 build 62942)


//////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2019 PANGO MICROSYSTEMS, INC
// ALL RIGHTS RESERVED.
//
// THE SOURCE CODE CONTAINED HEREIN IS PROPRIETARY TQ PANGO MICROSYSTEMS, INC.
// IT SHALL NOT BE REPRODUCED OR DISCLOSED IN WHOLE OR IN PART OR USED BY
// PARTIES WITHOUT WRITTEN AUTHORIZATION FROM THE OWNER.
//
/////////////////////////////////////////////////////////////////////////////
// Revision:1.0(initial)
//
//////////////////////////////////////////////////////////////////////////////

`timescale 1ps/1ps

module ddr_test_top_tb;


localparam real CLKIN_FREQ  = 50.0;

localparam DDR_TYPE   = "DDR3";    

localparam DATA_WIDTH = "16BIT";
 

localparam PLL_REFCLK_IN_PERIOD = 1000000 / CLKIN_FREQ;

reg                                pll_refclk_in;
reg                                bu_top_rst_n;
reg                                grs_n;

wire                               pad_loop_in    ;
wire                               pad_loop_in_h  ;

wire                               pad_rstn_ch0   ;
wire                               pad_ddr_clk_w  ;
wire                               pad_ddr_clkn_w ;
wire                               pad_csn_ch0    ;
wire [15:0]                        pad_addr_ch0   ;
wire [16-1:0]                      pad_dq_ch0     ;
wire [16/8-1:0]                    pad_dqs_ch0    ;
wire [16/8-1:0]                    pad_dqsn_ch0   ;
wire [16/8-1:0]                    pad_dm_rdqs_ch0;
wire                               pad_cke_ch0    ;
wire                               pad_odt_ch0    ;
wire                               pad_rasn_ch0   ;
wire                               pad_casn_ch0   ;
wire                               pad_wen_ch0    ;
wire [2:0]                         pad_ba_ch0     ;



ipsl_hmic_h_top_test u_ipsl_hmic_h_top_test(
    .top_rst_n                              (bu_top_rst_n          ),
    .pll_refclk_in                          (pll_refclk_in         ),
                                        
    .ddr_init_done                          (ddr_init_done),
    .ddrphy_rst_done                        (),
    .pll_lock                               (pll_lock              ),
    .clk_led                                (clk_led               ),

    .pad_loop_in                            (pad_loop_in  ),
    .pad_loop_in_h                          (pad_loop_in_h  ),
    .pad_rstn_ch0                           (pad_rstn_ch0  ),
    .pad_ddr_clk_w                          (pad_ddr_clk_w  ),
    .pad_ddr_clkn_w                         (pad_ddr_clkn_w  ),
    .pad_csn_ch0                            (pad_csn_ch0  ),
    .pad_addr_ch0                           (pad_addr_ch0  ),
    .pad_dq_ch0                             (pad_dq_ch0  ),
    .pad_dqs_ch0                            (pad_dqs_ch0  ),
    .pad_dqsn_ch0                           (pad_dqsn_ch0  ),
    .pad_dm_rdqs_ch0                        (pad_dm_rdqs_ch0  ),
    .pad_cke_ch0                            (pad_cke_ch0  ),
    .pad_odt_ch0                            (pad_odt_ch0  ),
    .pad_rasn_ch0                           (pad_rasn_ch0  ),
    .pad_casn_ch0                           (pad_casn_ch0  ),
    .pad_wen_ch0                            (pad_wen_ch0  ),
    .pad_ba_ch0                             (pad_ba_ch0  ),
    .pad_loop_out                           (pad_loop_in  ),
    .pad_loop_out_h                         (pad_loop_in_h  ),
    .err_flag                               ()
);

reg pad_ddr_clk_w_dly, pad_ddr_clkn_w_dly;

always @ (*) begin
    pad_ddr_clk_w_dly <= #50 pad_ddr_clk_w;
    pad_ddr_clkn_w_dly <= #50 pad_ddr_clkn_w;
end

generate
if ((DDR_TYPE == "DDR3")&&(DATA_WIDTH == "16BIT")) begin
ddr3     mem_core (

    .rst_n                           (pad_rstn_ch0  ),
    .ck                              (pad_ddr_clk_w_dly  ),
    .ck_n                            (pad_ddr_clkn_w_dly  ),
    .cs_n                            (pad_csn_ch0  ),
//    .addr                            (pad_addr_ch0[ADDR_BITS-1:0]  ),
    .addr                            (pad_addr_ch0  ),
    .dq                              (pad_dq_ch0  ),
    .dqs                             (pad_dqs_ch0  ),
    .dqs_n                           (pad_dqsn_ch0  ),
    .dm_tdqs                         (pad_dm_rdqs_ch0  ),
    .tdqs_n                          (  ),
    .cke                             (pad_cke_ch0  ),
    .odt                             (pad_odt_ch0  ),
    .ras_n                           (pad_rasn_ch0  ),
    .cas_n                           (pad_casn_ch0  ),
    .we_n                            (pad_wen_ch0  ),
    .ba                              (pad_ba_ch0  )

);
end
else if ((DDR_TYPE == "DDR3")&&(DATA_WIDTH == "8BIT")) begin
ddr3     mem_core (

    .rst_n                           (pad_rstn_ch0  ),
    .ck                              (pad_ddr_clk_w_dly  ),
    .ck_n                            (pad_ddr_clkn_w_dly  ),
    .cs_n                            (pad_csn_ch0  ),
//    .addr                            (pad_addr_ch0[ADDR_BITS-1:0]  ),
    .addr                            (pad_addr_ch0  ),
    .dq                              (pad_dq_ch0[7:0]  ),
    .dqs                             (pad_dqs_ch0[0]  ),
    .dqs_n                           (pad_dqsn_ch0[0]  ),
    .dm_tdqs                         (pad_dm_rdqs_ch0[0]  ),
    .tdqs_n                          (  ),
    .cke                             (pad_cke_ch0  ),
    .odt                             (pad_odt_ch0  ),
    .ras_n                           (pad_rasn_ch0  ),
    .cas_n                           (pad_casn_ch0  ),
    .we_n                            (pad_wen_ch0  ),
    .ba                              (pad_ba_ch0  )

);
end
else if ((DDR_TYPE == "DDR2")&&(DATA_WIDTH == "16BIT")) begin
ddr2 mem_core (         
    .ck                              (pad_ddr_clk_w_dly  ), 
    .ck_n                            (pad_ddr_clkn_w_dly  ),
    .cs_n                            (pad_csn_ch0  ),       
    .addr                            (pad_addr_ch0  ),      
    .dq                              (pad_dq_ch0  ),        
    .dqs                             (pad_dqs_ch0  ),       
    .dqs_n                           (pad_dqsn_ch0  ),      
    .dm_rdqs                         (pad_dm_rdqs_ch0  ),   
    .rdqs_n                          (  ),                  
    .cke                             (pad_cke_ch0  ),       
    .odt                             (pad_odt_ch0  ),       
    .ras_n                           (pad_rasn_ch0  ),      
    .cas_n                           (pad_casn_ch0  ),      
    .we_n                            (pad_wen_ch0  ),       
    .ba                              (pad_ba_ch0  )         
);
end
else if ((DDR_TYPE == "DDR2")&&(DATA_WIDTH == "8BIT")) begin
ddr2 mem_core (         
    .ck                              (pad_ddr_clk_w_dly  ), 
    .ck_n                            (pad_ddr_clkn_w_dly  ),
    .cs_n                            (pad_csn_ch0  ),       
    .addr                            (pad_addr_ch0  ),      
    .dq                              (pad_dq_ch0[7:0]  ),        
    .dqs                             (pad_dqs_ch0[0]  ),       
    .dqs_n                           (pad_dqsn_ch0[0]  ),      
    .dm_rdqs                         (pad_dm_rdqs_ch0[0]  ),   
    .rdqs_n                          (  ),                  
    .cke                             (pad_cke_ch0  ),       
    .odt                             (pad_odt_ch0  ),       
    .ras_n                           (pad_rasn_ch0  ),      
    .cas_n                           (pad_casn_ch0  ),      
    .we_n                            (pad_wen_ch0  ),       
    .ba                              (pad_ba_ch0  )         
);
end
else if ((DDR_TYPE == "LPDDR")&&(DATA_WIDTH == "16BIT")) begin 
mobile_ddr mem_core ( 
   .Clk                            (pad_ddr_clk_w_dly), 
   .Clk_n                          (pad_ddr_clkn_w_dly),
   .Cke                            (pad_cke_ch0),
   .Cs_n                           (pad_csn_ch0),
   .Ras_n                          (pad_rasn_ch0),
   .Cas_n                          (pad_casn_ch0),
   .We_n                           (pad_wen_ch0),
   .Addr                           (pad_addr_ch0[13:0]),
   .Ba                             (pad_ba_ch0[1:0]),
   .Dq                             (pad_dq_ch0),
   .Dqs                            ({pad_dqs_ch0[1],pad_dqs_ch0[0]}),
   .Dm                             (pad_dm_rdqs_ch0)
);
end
else if ((DDR_TYPE == "LPDDR")&&(DATA_WIDTH == "8BIT")) begin 
mobile_ddr mem_core ( 
   .Clk                            (pad_ddr_clk_w_dly), 
   .Clk_n                          (pad_ddr_clkn_w_dly),
   .Cke                            (pad_cke_ch0),
   .Cs_n                           (pad_csn_ch0),
   .Ras_n                          (pad_rasn_ch0),
   .Cas_n                          (pad_casn_ch0),
   .We_n                           (pad_wen_ch0),
   .Addr                           (pad_addr_ch0[13:0]),
   .Ba                             (pad_ba_ch0[1:0]),
   .Dq                             (pad_dq_ch0[7:0]),
   .Dqs                            (pad_dqs_ch0[0]),
   .Dm                             (pad_dm_rdqs_ch0[0])
);
end
endgenerate 

/********************clk and init******************/

always #(PLL_REFCLK_IN_PERIOD / 2)  pll_refclk_in = ~pll_refclk_in;

initial begin

#1 
pll_refclk_in = 0;

//default input from keyboard
bu_top_rst_n = 1'b1;

end
/*******************end of clk and init*******************/


//GTP_GRS I_GTP_GRS(
GTP_GRS GRS_INST(
		.GRS_N (grs_n)
	);
initial begin
grs_n = 1'b0;
#5000 grs_n = 1'b1;
end

initial begin

//reset the bu_top
#10000 bu_top_rst_n = 1'b0;
#50000 bu_top_rst_n = 1'b1;
$display("%t keyboard reset sequence finished!", $time);

@ (posedge ddr_init_done);
$display("%t ddr_init_done is high now!", $time);
#100000000;
$finish;
end

initial 
begin
 $fsdbDumpfile("ddr_test_top_tb.fsdb");
 $fsdbDumpvars(0,"ddr_test_top_tb");
end

endmodule
