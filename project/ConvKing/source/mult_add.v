`timescale 1ns/1ps

module mult_add(
           input sys_clk_i,
           input sys_rst_n_i,
           input sys_is_add_i,
           input [15: 0] am_num_A0_i,
           input [15: 0] am_num_A1_i,
           input [15: 0] am_num_B0_i,
           input [15: 0] am_num_B1_i,
           output [15: 0] am_mult_o
       );

mult_add_gen mult_add_inst_0(
    .ce(1'b1),
    .rst(sys_rst_n_i),
    .clk(sys_clk_i),
    .a0(am_num_A0_i),
    .a1(am_num_A1_i),
    .b0(am_num_B0_i),
    .b1(am_num_B1_i),
    .addsub(sys_is_add_i),
    .p(am_mult_o)
);

endmodule
