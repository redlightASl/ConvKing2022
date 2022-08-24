`timescale 1ns/1ps

module yolo_top #(
           parameter YOLO_INPUT_BITWIDTH = 16,
           parameter YOLO_OUTPUT_BITWIDTH = 16
       )
       (
           //system
           input sys_clk,
           input sys_rst_n,
           //input data
           input [YOLO_INPUT_BITWIDTH - 1: 0] rgb_i,
           //input ctrl data
           input hsync_i,
           input vsync_i,
           input de_i,
           //output data
           output [YOLO_OUTPUT_BITWIDTH - 1: 0] data_out,
           //output ctrl data
           output yolo_hsync,
           output yolo_vsync,
           output yolo_de
       );


mobilenet_top mobile_net(

);


yolo_pre yolo_pre_inst(

         );



yolo_aft u_yolo_aft(
             //ports
             .pixelclk ( ),
             .reset_n ( ),
             .red_en ( ),
             .grenn_en ( ),
             .blue_en ( ),
             .i_rgb ( ),
             .i_hsync ( ),
             .i_vsync ( ),
             .i_de ( ),
             .hcount ( ),
             .vcount ( ),
             .hcount_l ( ),
             .hcount_r ( ),
             .vcount_l ( ),
             .vcount_r ( ),
             .o_rgb ( ),
             .o_hsync ( ),
             .o_vsync ( ),
             .o_de ( )
         );

assign yolo_hsync = hsync_i;
assign yolo_vsync = vsync_i;
assign yolo_de = de_i;

// assign data_out[YOLO_OUTPUT_BITWIDTH - 1: 0] = {rgb_i[YOLO_INPUT_BITWIDTH - 1-: 5], rgb_i[YOLO_INPUT_BITWIDTH - 1-: 5], rgb_i[YOLO_INPUT_BITWIDTH - 1-: 5]};
assign data_out[YOLO_OUTPUT_BITWIDTH - 1: 0] = rgb_i[YOLO_INPUT_BITWIDTH - 1: 0];


endmodule
