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

module ipsl_hmic_h_top_test (
    input                                top_rst_n              ,
    input                                pll_refclk_in          ,                                                               
    output reg                           clk_led                ,
    output                               pll_lock               ,
    output                               ddr_init_done          ,
    output                               ddrphy_rst_done        ,                                                                                                                          
    input                                pad_loop_in            ,
    input                                pad_loop_in_h          ,
    output                               pad_rstn_ch0           ,
    output                               pad_ddr_clk_w          ,
    output                               pad_ddr_clkn_w         ,
    output                               pad_csn_ch0            ,
    output [15:0]                        pad_addr_ch0           ,
    inout  [16-1:0]                      pad_dq_ch0             ,
    inout  [16/8-1:0]                    pad_dqs_ch0            ,
    inout  [16/8-1:0]                    pad_dqsn_ch0           ,
    output [16/8-1:0]                    pad_dm_rdqs_ch0        ,
    output                               pad_cke_ch0            ,
    output                               pad_odt_ch0            ,
    output                               pad_rasn_ch0           ,
    output                               pad_casn_ch0           ,
    output                               pad_wen_ch0            ,
    output [2:0]                         pad_ba_ch0             ,
    output                               pad_loop_out           ,
    output                               pad_loop_out_h         ,
    output                               err_flag                                                                      
);


localparam MEM_ROW_ADDRESS = 14;

localparam MEM_COLUMN_ADDRESS = 10;

localparam MEM_BANK_ADDRESS = 3;


`ifdef SIMULATION
localparam MEM_SPACE_AW = 10; //to reduce simulation time
`else
localparam MEM_SPACE_AW = MEM_ROW_ADDRESS + MEM_COLUMN_ADDRESS + MEM_BANK_ADDRESS;
`endif

localparam CTRL_ADDR_WIDTH = MEM_ROW_ADDRESS + MEM_COLUMN_ADDRESS + MEM_BANK_ADDRESS;

localparam TH_1S = 27'd33000000;

reg [26:0] cnt; 

    wire  [32-1:0]    axi_awaddr          ;
    wire  [7:0]       axi_awid            ;
    wire  [7:0]       axi_awlen           ;
    wire  [2:0]       axi_awsize          ;
    wire  [1:0]       axi_awburst         ;
    wire              axi_awlock          ;
    wire              axi_awready         ;
    wire              axi_awvalid         ;
    wire [3:0]        axi_awqos           ;
    wire              axi_awurgent        ;
    wire              axi_awpoison        ;
          
    wire  [64-1:0]   axi_wdata           ;
    wire  [8-1:0]    axi_wstrb           ;
    
    wire              axi_wvalid          ;
    wire              axi_wready          ;
    wire              axi_wlast           ;
    
    wire [7:0]         axi_bid            ;
    wire [1:0]         axi_bresp          ;
    wire               axi_bvalid         ;
    wire               axi_bready         ;

    wire  [32-1:0]    axi_araddr          ;
    wire  [7:0]       axi_arid            ;
    wire  [7:0]       axi_arlen           ;
    wire [2:0]        axi_arsize          ;
    wire [1:0]        axi_arburst         ;
    wire              axi_arlock          ;
    wire [3:0]        axi_arqos           ;
    wire              axi_arpoison        ;
    wire              axi_arurgent        ;
    wire              axi_arready ;
    wire              axi_arvalid ;
 
    wire   [64-1:0]  axi_rdata /* synthesis syn_keep = 1 */;
     
    wire   [7:0]      axi_rid ;
    wire              axi_rlast ;
    wire              axi_rvalid /* synthesis syn_keep = 1 */;
    wire              axi_rready;
    wire [1:0]        axi_rresp ;
    wire              axi_csysreq ;
    wire              axi_csysack ;
    wire              axi_cactive ;      

    wire              pll_pclk;                                                         

   wire [CTRL_ADDR_WIDTH-1:0] random_rw_addr ;  
   wire [3:0] random_axi_id   ; 
   wire [3:0] random_axi_len  ;                      
   wire init_start            ;
   wire init_done             ;
   wire write_en              ;
   wire write_done_p          ;
   wire read_en               ;
   wire read_done_p           ;

wire axi_clk;

always@(posedge axi_clk or negedge top_rst_n)
begin
   if (!top_rst_n)
      cnt <= 27'd0;       
   else if ( cnt >= TH_1S )   
      cnt <= 27'd0;
   else   
      cnt <= cnt + 27'd1;
end

always @(posedge axi_clk or negedge top_rst_n)
begin
   if (!top_rst_n)  
      clk_led <= 1'd1;
   else if (cnt >= TH_1S)
      clk_led <= ~clk_led;
end


test_main_ctrl #(
 .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH),
 .MEM_DQ_WIDTH    (16),
 .MEM_SPACE_AW    (MEM_SPACE_AW)
) u_test_main_ctrl
(
 .random_rw_addr        (random_rw_addr   ),
 .random_axi_id         (random_axi_id    ),
 .random_axi_len        (random_axi_len   ),
 .clk                   (axi_clk          ),
 .rst_n                 (top_rst_n        ), 
 .ddrc_init_done        (ddr_init_done    ),
 .init_start            (init_start       ),
 .init_done             (init_done        ),
 .write_en              (write_en         ),
 .write_done_p          (write_done_p     ),
 .read_en               (read_en          ),
 .read_done_p           (read_done_p      )
);


test_wr_ctrl_64bit #(
 .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH),
 .MEM_DQ_WIDTH    (16),
 .MEM_SPACE_AW    (MEM_SPACE_AW)
) u_test_wr_ctrl
(
 .clk                 (axi_clk         ),
 .rst_n               (top_rst_n       ),    
 .init_start          (init_start      ),
 .write_en            (write_en        ),
 .write_done_p        (write_done_p    ),
 .init_done           (init_done       ),
 .random_rw_addr      (random_rw_addr  ),     
 .random_axi_id       (random_axi_id   ),
 .random_axi_len      (random_axi_len  ),
 .data_pattern_01     (1'b0 ),
 .random_data_en      (1'b0 ),
 .axi_awaddr          (axi_awaddr      ),
 .axi_awid            (axi_awid        ),
 .axi_awlen           (axi_awlen       ),
 .axi_awsize          (axi_awsize      ),
 .axi_awburst         (axi_awburst     ),
 .axi_awlock          (axi_awlock      ),
 .axi_awready         (axi_awready     ),
 .axi_awvalid         (axi_awvalid     ),
 .axi_awqos           (axi_awqos       ),
 .axi_awurgent        (axi_awurgent    ),
 .axi_awpoison        (axi_awpoison    ),          
 .axi_wdata           (axi_wdata       ),
 .axi_wstrb           (axi_wstrb       ),
 .axi_wvalid          (axi_wvalid      ),
 .axi_wready          (axi_wready      ),
 .axi_wlast           (axi_wlast       ),    
 .axi_bid             (axi_bid         ),
 .axi_bresp           (axi_bresp       ),
 .axi_bvalid          (axi_bvalid      ),
 .axi_bready          (axi_bready      )        
);

test_rd_ctrl_64bit #(
 .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH),
 .MEM_DQ_WIDTH    (16),
 .MEM_SPACE_AW    (MEM_SPACE_AW)
)u_test_rd_ctrl
(
 .random_rw_addr      (random_rw_addr  ),
 .random_axi_id       (random_axi_id   ),
 .random_axi_len      (random_axi_len  ),
 .clk                 (axi_clk         ),
 .rst_n               (top_rst_n       ),   
 .read_en             (read_en         ),
 .data_pattern_01     (1'b0            ),
 .read_double_en      (1'b0            ),   
 .read_done_p         (read_done_p     ),   
 .axi_araddr          (axi_araddr      ),
 .axi_arid            (axi_arid        ),
 .axi_arlen           (axi_arlen       ),
 .axi_arsize          (axi_arsize      ),
 .axi_arburst         (axi_arburst     ),
 .axi_arlock          (axi_arlock      ),
 .axi_arqos           (axi_arqos       ),
 .axi_arpoison        (axi_arpoison    ),
 .axi_arurgent        (axi_arurgent    ),
 .axi_arready         (axi_arready     ),
 .axi_arvalid         (axi_arvalid     ),
 .axi_rdata           (axi_rdata       ),
 .axi_rid             (axi_rid         ),
 .axi_rlast           (axi_rlast       ),
 .axi_rvalid          (axi_rvalid      ),
 .axi_rready          (axi_rready      ),
 .axi_rresp           (axi_rresp       ),
 .err_cnt             (                ),   
 .err_flag_led        (err_flag    )
);
ddr3_core u_ipsl_hmic_h_top (
    .pll_refclk_in        (pll_refclk_in ),
    .top_rst_n            (top_rst_n     ),
      
    .pll_aclk_0           (              ),
    .pll_aclk_1           (axi_clk       ),
    .pll_aclk_2           (              ),
    
    .pll_lock             (pll_lock      ),
    .ddrphy_rst_done      (ddrphy_rst_done),
    .ddrc_init_done       (ddr_init_done ),
    .ddrc_rst         (0),    
      
    .areset_1         (0),               
    .aclk_1           (axi_clk),                                                        
    .awid_1           (axi_awid),       
    .awaddr_1         (axi_awaddr),     
    .awlen_1          (axi_awlen),      
    .awsize_1         (axi_awsize),     
    .awburst_1        (axi_awburst),    
    .awlock_1         (axi_awlock),                       
    .awvalid_1        (axi_awvalid),    
    .awready_1        (axi_awready),    
    .awurgent_1       (axi_awurgent),   
    .awpoison_1       (axi_awpoison),                     
    .wdata_1          (axi_wdata),      
    .wstrb_1          (axi_wstrb),      
    .wlast_1          (axi_wlast),      
    .wvalid_1         (axi_wvalid),     
    .wready_1         (axi_wready),                       
    .bid_1            (axi_bid),        
    .bresp_1          (axi_bresp),      
    .bvalid_1         (axi_bvalid),     
    .bready_1         (axi_bready),                                    
    .arid_1           (axi_arid     ),  
    .araddr_1         (axi_araddr   ),  
    .arlen_1          (axi_arlen    ),  
    .arsize_1         (axi_arsize   ),  
    .arburst_1        (axi_arburst  ),  
    .arlock_1         (axi_arlock   ),                      
    .arvalid_1        (axi_arvalid  ),  
    .arready_1        (axi_arready  ),  
    .arpoison_1       (axi_arpoison ),                      
    .rid_1            (axi_rid      ),  
    .rdata_1          (axi_rdata    ),  
    .rresp_1          (axi_rresp    ),  
    .rlast_1          (axi_rlast    ),  
    .rvalid_1         (axi_rvalid   ),  
    .rready_1         (axi_rready   ),  
    .arurgent_1       (axi_arurgent ),                
    .csysreq_1        (1'b1),               
    .csysack_1        (),           
    .cactive_1        (), 
          
    .csysreq_ddrc     (1'b1),
    .csysack_ddrc     (),
    .cactive_ddrc     (),
             
    .pad_loop_in           (pad_loop_in),
    .pad_loop_in_h         (pad_loop_in_h),
    .pad_rstn_ch0          (pad_rstn_ch0),
    .pad_ddr_clk_w         (pad_ddr_clk_w),
    .pad_ddr_clkn_w        (pad_ddr_clkn_w),
    .pad_csn_ch0           (pad_csn_ch0),
    .pad_addr_ch0          (pad_addr_ch0),
    .pad_dq_ch0            (pad_dq_ch0),
    .pad_dqs_ch0           (pad_dqs_ch0),
    .pad_dqsn_ch0          (pad_dqsn_ch0),
    .pad_dm_rdqs_ch0       (pad_dm_rdqs_ch0),
    .pad_cke_ch0           (pad_cke_ch0),
    .pad_odt_ch0           (pad_odt_ch0),
    .pad_rasn_ch0          (pad_rasn_ch0),
    .pad_casn_ch0          (pad_casn_ch0),
    .pad_wen_ch0           (pad_wen_ch0),
    .pad_ba_ch0            (pad_ba_ch0),
    .pad_loop_out          (pad_loop_out),
    .pad_loop_out_h        (pad_loop_out_h)                                
);                                      

endmodule
