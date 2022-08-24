`timescale 1ns/1ps
module knn_top_tb();

reg clk_en;
reg reset_n;
reg knn_en;

reg [9:0] postion_lu_x;
reg [9:0] postion_lu_y;
reg [9:0] postion_rd_x;
reg [9:0] postion_rd_y;

wire [3:0] knn_result;
wire knn_outflag;

initial begin
    clk_en=0;
    forever #10 clk_en=~clk_en;
end


initial begin
    reset_n=0;
    #40 reset_n=1;
    //#20000 reset_n=0;
end

initial begin
    postion_lu_x=0;
    postion_lu_y=0;
    postion_rd_x=4;
    postion_rd_y=4;
end

initial begin
    knn_en=0;
    #30 knn_en=1;
end

knn_top inst(
    .clk_en     (clk_en),
    .reset_n    (reset_n),
    .knn_en     (knn_en),

    .postion_lu_x   (postion_lu_x),
    .postion_lu_y   (postion_lu_y),
    .postion_rd_x   (postion_rd_x),
    .postion_rd_y   (postion_rd_y),
    
    .knn_result     (knn_result),
    .knn_outflag    (knn_outflag)
);




endmodule