`timescale 1ns/1ps

module serdes_4b_10to1 (
	input          clk,           // clock input
	input          clkx5,         // 5x clock input
	input [9:0]    datain_0,      // input data for serialisation
	input [9:0]    datain_1,      // input data for serialisation
	input [9:0]    datain_2,      // input data for serialisation
	input [9:0]    datain_3,      // input data for serialisation
	output         dataout_0_p,   // out DDR data
	output         dataout_0_n,   // out DDR data
	output         dataout_1_p,   // out DDR data
	output         dataout_1_n,   // out DDR data
	output         dataout_2_p,   // out DDR data
	output         dataout_2_n,   // out DDR data
	output         dataout_3_p,   // out DDR data
	output         dataout_3_n    // out DDR data
  ) ;  
wire         padt0_p     ;	
wire         padt1_p     ;
wire         padt2_p     ;
wire         padt3_p     ;
wire  [3:0]  stxd_rgm_p  ; 
wire         padt0_n     ;	
wire         padt1_n     ;
wire         padt2_n     ;
wire         padt3_n     ;
wire  [3:0]  stxd_rgm_n  ;  
reg [2:0] TMDS_mod5 = 0;  // modulus 5 counter

reg [4:0] TMDS_shift_0h = 0, TMDS_shift_0l = 0;
reg [4:0] TMDS_shift_1h = 0, TMDS_shift_1l = 0;
reg [4:0] TMDS_shift_2h = 0, TMDS_shift_2l = 0;
reg [4:0] TMDS_shift_3h = 0, TMDS_shift_3l = 0;

wire [4:0] TMDS_0_l = {datain_0[9],datain_0[7],datain_0[5],datain_0[3],datain_0[1]};
wire [4:0] TMDS_0_h = {datain_0[8],datain_0[6],datain_0[4],datain_0[2],datain_0[0]};

wire [4:0] TMDS_1_l = {datain_1[9],datain_1[7],datain_1[5],datain_1[3],datain_1[1]};
wire [4:0] TMDS_1_h = {datain_1[8],datain_1[6],datain_1[4],datain_1[2],datain_1[0]};

wire [4:0] TMDS_2_l = {datain_2[9],datain_2[7],datain_2[5],datain_2[3],datain_3[1]};
wire [4:0] TMDS_2_h = {datain_2[8],datain_2[6],datain_2[4],datain_2[2],datain_3[0]};

wire [4:0] TMDS_3_l = {datain_3[9],datain_3[7],datain_3[5],datain_3[3],datain_3[1]};
wire [4:0] TMDS_3_h = {datain_3[8],datain_3[6],datain_3[4],datain_3[2],datain_3[0]};

always @(posedge clkx5)
begin
	TMDS_shift_0h  <= TMDS_mod5[2] ? TMDS_0_h : TMDS_shift_0h[4:1];
	TMDS_shift_0l  <= TMDS_mod5[2] ? TMDS_0_l : TMDS_shift_0l[4:1];
	TMDS_shift_1h  <= TMDS_mod5[2] ? TMDS_1_h : TMDS_shift_1h[4:1];
	TMDS_shift_1l  <= TMDS_mod5[2] ? TMDS_1_l : TMDS_shift_1l[4:1];
	TMDS_shift_2h  <= TMDS_mod5[2] ? TMDS_2_h : TMDS_shift_2h[4:1];
	TMDS_shift_2l  <= TMDS_mod5[2] ? TMDS_2_l : TMDS_shift_2l[4:1];
	TMDS_shift_3h  <= TMDS_mod5[2] ? TMDS_3_h : TMDS_shift_3h[4:1];
	TMDS_shift_3l  <= TMDS_mod5[2] ? TMDS_3_l : TMDS_shift_3l[4:1];	
	TMDS_mod5 <= (TMDS_mod5[2]) ? 3'd0 : TMDS_mod5 + 3'd1;
end


GTP_OSERDES #(
 .OSERDES_MODE("ODDR"),  //"ODDR","OMDDR","OGSER4","OMSER4","OGSER7","OGSER8",OMSER8"
 .WL_EXTEND   ("FALSE"),     //"TRUE"; "FALSE"
 .GRS_EN      ("TRUE"),         //"TRUE"; "FALSE"
 .LRS_EN      ("TRUE"),          //"TRUE"; "FALSE"
 .TSDDR_INIT  (1'b0)         //1'b0;1'b1
) gtp_ogddr0(
   .DO    (stxd_rgm_p[3]),
   .TQ    (padt3_p),
   .DI    ({6'd0,TMDS_shift_3l[0],TMDS_shift_3h[0]}),
   .TI    (4'd0),
   .RCLK  (clkx5),
   .SERCLK(clkx5),
   .OCLK  (1'd0),
   .RST   (1'b0)
); 
GTP_OUTBUFT  gtp_outbuft0
(
    
    .I(stxd_rgm_p[3]),     
    .T(padt3_p)  ,
    .O(dataout_3_p)        
);

GTP_OSERDES #(
 .OSERDES_MODE("ODDR"),  //"ODDR","OMDDR","OGSER4","OMSER4","OGSER7","OGSER8",OMSER8"
 .WL_EXTEND   ("FALSE"),     //"TRUE"; "FALSE"
 .GRS_EN      ("TRUE"),         //"TRUE"; "FALSE"
 .LRS_EN      ("TRUE"),          //"TRUE"; "FALSE"
 .TSDDR_INIT  (1'b0)         //1'b0;1'b1
) gtp_ogddr1(
   .DO    (stxd_rgm_p[2]),
   .TQ    (padt2_p),
   .DI    ({6'd0,TMDS_shift_2l[0],TMDS_shift_2h[0]}),
   .TI    (4'd0),
   .RCLK  (clkx5),
   .SERCLK(clkx5),
   .OCLK  (1'd0),
   .RST   (1'b0)
); 
GTP_OUTBUFT  gtp_outbuft1
(
    
    .I(stxd_rgm_p[2]),     
    .T(padt2_p)  ,
    .O(dataout_2_p)        
);    
  
GTP_OSERDES #(
 .OSERDES_MODE("ODDR"),  //"ODDR","OMDDR","OGSER4","OMSER4","OGSER7","OGSER8",OMSER8"
 .WL_EXTEND   ("FALSE"),     //"TRUE"; "FALSE"
 .GRS_EN      ("TRUE"),         //"TRUE"; "FALSE"
 .LRS_EN      ("TRUE"),          //"TRUE"; "FALSE"
 .TSDDR_INIT  (1'b0)         //1'b0;1'b1
) gtp_ogddr2(
   .DO    (stxd_rgm_p[1]),
   .TQ    (padt1_p),
   .DI    ({6'd0,TMDS_shift_1l[0],TMDS_shift_1h[0]}),
   .TI    (4'd0),
   .RCLK  (clkx5),
   .SERCLK(clkx5),
   .OCLK  (1'd0),
   .RST   (1'b0)
); 
GTP_OUTBUFT  gtp_outbuft2
(
    
    .I(stxd_rgm_p[1]),     
    .T(padt1_p)  ,
    .O(dataout_1_p)        
); 
    
GTP_OSERDES #(
 .OSERDES_MODE("ODDR"),  //"ODDR","OMDDR","OGSER4","OMSER4","OGSER7","OGSER8",OMSER8"
 .WL_EXTEND   ("FALSE"),     //"TRUE"; "FALSE"
 .GRS_EN      ("TRUE"),         //"TRUE"; "FALSE"
 .LRS_EN      ("TRUE"),          //"TRUE"; "FALSE"
 .TSDDR_INIT  (1'b0)         //1'b0;1'b1
) gtp_ogddr3(
   .DO    (stxd_rgm_p[0]),
   .TQ    (padt0_p),
   .DI    ({6'd0,TMDS_shift_0l[0],TMDS_shift_0h[0]}),
   .TI    (4'd0),
   .RCLK  (clkx5),
   .SERCLK(clkx5),
   .OCLK  (1'd0),
   .RST   (1'b0)
); 
GTP_OUTBUFT  gtp_outbuft3
(
    
    .I(stxd_rgm_p[0]),     
    .T(padt0_p)  ,
    .O(dataout_0_p)        
);             
   
GTP_OSERDES #(
 .OSERDES_MODE("ODDR"),  //"ODDR","OMDDR","OGSER4","OMSER4","OGSER7","OGSER8",OMSER8"
 .WL_EXTEND   ("FALSE"),     //"TRUE"; "FALSE"
 .GRS_EN      ("TRUE"),         //"TRUE"; "FALSE"
 .LRS_EN      ("TRUE"),          //"TRUE"; "FALSE"
 .TSDDR_INIT  (1'b0)         //1'b0;1'b1
) gtp_ogddr4(
   .DO    (stxd_rgm_n[3]),
   .TQ    (padt3_n),
   .DI    ({6'd0,~TMDS_shift_3l[0],~TMDS_shift_3h[0]}),
   .TI    (4'd0),
   .RCLK  (clkx5),
   .SERCLK(clkx5),
   .OCLK  (1'd0),
   .RST   (1'b0)
); 
GTP_OUTBUFT  gtp_outbuft4
(
    
    .I(stxd_rgm_n[3]),     
    .T(padt3_n)  ,
    .O(dataout_3_n)        
);
GTP_OSERDES #(
 .OSERDES_MODE("ODDR"),  //"ODDR","OMDDR","OGSER4","OMSER4","OGSER7","OGSER8",OMSER8"
 .WL_EXTEND   ("FALSE"),     //"TRUE"; "FALSE"
 .GRS_EN      ("TRUE"),         //"TRUE"; "FALSE"
 .LRS_EN      ("TRUE"),          //"TRUE"; "FALSE"
 .TSDDR_INIT  (1'b0)         //1'b0;1'b1
) gtp_ogddr5(
   .DO    (stxd_rgm_n[2]),
   .TQ    (padt2_n),
   .DI    ({6'd0,~TMDS_shift_2l[0],~TMDS_shift_2h[0]}),
   .TI    (4'd0),
   .RCLK  (clkx5),
   .SERCLK(clkx5),
   .OCLK  (1'd0),
   .RST   (1'b0)
); 
GTP_OUTBUFT  gtp_outbuft5
(
    
    .I(stxd_rgm_n[2]),     
    .T(padt2_n)  ,
    .O(dataout_2_n)        
);    

GTP_OSERDES #(
 .OSERDES_MODE("ODDR"),  //"ODDR","OMDDR","OGSER4","OMSER4","OGSER7","OGSER8",OMSER8"
 .WL_EXTEND   ("FALSE"),     //"TRUE"; "FALSE"
 .GRS_EN      ("TRUE"),         //"TRUE"; "FALSE"
 .LRS_EN      ("TRUE"),          //"TRUE"; "FALSE"
 .TSDDR_INIT  (1'b0)         //1'b0;1'b1
) gtp_ogddr6(
   .DO    (stxd_rgm_n[1]),
   .TQ    (padt1_n),
   .DI    ({6'd0,~TMDS_shift_1l[0],~TMDS_shift_1h[0]}),
   .TI    (4'd0),
   .RCLK  (clkx5),
   .SERCLK(clkx5),
   .OCLK  (1'd0),
   .RST   (1'b0)
); 
GTP_OUTBUFT  gtp_outbuft6
(
    
    .I(stxd_rgm_n[1]),     
    .T(padt1_n)  ,
    .O(dataout_1_n)        
); 
    
GTP_OSERDES #(
 .OSERDES_MODE("ODDR"),  //"ODDR","OMDDR","OGSER4","OMSER4","OGSER7","OGSER8",OMSER8"
 .WL_EXTEND   ("FALSE"),     //"TRUE"; "FALSE"
 .GRS_EN      ("TRUE"),         //"TRUE"; "FALSE"
 .LRS_EN      ("TRUE"),          //"TRUE"; "FALSE"
 .TSDDR_INIT  (1'b0)         //1'b0;1'b1
) gtp_ogddr7(
   .DO    (stxd_rgm_n[0]),
   .TQ    (padt0_n),
   .DI    ({6'd0,~TMDS_shift_0l[0],~TMDS_shift_0h[0]}),
   .TI    (4'd0),
   .RCLK  (clkx5),
   .SERCLK(clkx5),
   .OCLK  (1'd0),
   .RST   (1'b0)
); 
GTP_OUTBUFT  gtp_outbuft7
(
    
    .I(stxd_rgm_n[0]),     
    .T(padt0_n)  ,
    .O(dataout_0_n)        
);             

	
endmodule
