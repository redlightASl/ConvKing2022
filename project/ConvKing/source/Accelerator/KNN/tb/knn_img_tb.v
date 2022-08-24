module knn_img_tb();

reg clk_en;
reg reset_n;

reg dic_end;
reg knn_en;

reg [9:0] postion_lu_x;
reg [9:0] postion_lu_y;
reg [9:0] postion_rd_x;
reg [9:0] postion_rd_y;

wire [9:0] i_o;
wire [9:0] j_o;

wire [3:0] cnt_w_o;
wire [3:0] cnt_h_o;
wire [9:0] wid_center_o;
wire [9:0] hei_center_o;
wire       knn_fin_o; 

initial begin
    dic_end=0;
    knn_en=0;
    #200;
    dic_end=1;
    #600 knn_en=1;
    #200 knn_en=0;
end

initial begin
    clk_en=0;
    forever #100 clk_en=~clk_en;
end

initial begin
    reset_n=0;
    #400 reset_n=1;
    #20000 reset_n=0;
end

initial begin
    postion_lu_x=100;
    postion_lu_y=400;
    postion_rd_x=300;
    postion_rd_y=200;
end

initial begin
    knn_en=0;
    #300 knn_en=1;
end

knn_img inst(
    .clk_en     (clk_en),
    .reset_n    (reset_n),

    .dic_end    (dic_end),
    .knn_en     (knn_en),

    .postion_lu_x   (postion_lu_x),
    .postion_lu_y   (postion_lu_y),
    .postion_rd_x   (postion_rd_x),
    .postion_rd_y   (postion_rd_y),

    .i_o            (i_o),
    .j_o            (j_o),

    .cnt_w_o        (cnt_w_o),
    .cnt_h_o        (cnt_h_o),
    .wid_center_o   (wid_center_o),
    .hei_center_o   (hei_center_o),
    .knn_fin_o      (knn_fin_o)
);

endmodule