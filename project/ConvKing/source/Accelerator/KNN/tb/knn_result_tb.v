`timescale 1ns/1ps
module knn_result_tb();

reg clk_en;
reg reset_n;
reg dic_go;

reg [13:0]  distance ;
reg [3:0]   m ;

reg dic_end;
reg dic_end_q;
reg knn_fin;
   

wire [13:0] min_o;
wire [3:0]  min_p_o;
wire [3:0]  max_p_o;
wire [5:0]  max_o;   
wire [3:0]  max_cnt_o;
wire q0_o;
wire q1_o;
wire fin_o;
wire stop_o;

wire [5:0]    color_cnt_0;
wire [5:0]    color_cnt_1;
wire [5:0]    color_cnt_2;
wire [5:0]    color_cnt_3;
wire [5:0]    color_cnt_4;

wire [3:0] knn_resultf;
wire out_flag;

initial begin
    clk_en=0;
    forever #100 clk_en=~clk_en;
end

initial begin
    reset_n=0;
    #400 reset_n=1;
    //#8000 reset_n=0;
end

initial begin
    dic_go=0;
    #500    dic_go=1;
end

initial begin
    knn_fin=0;
    dic_end=0;
    dic_end_q=0;
    m=0;
    distance={$random} % 1024;
    #600;
    repeat(16) begin
    repeat(4) begin
        @(posedge clk_en);
        #1;
        m=m+1;
        dic_end_q=0;
        distance={$random} % 1024;
    end
    dic_end=1;
    @(posedge clk_en);
    #1;
    dic_end_q=1;
    @(posedge clk_en);
    #1;
    dic_end=0;
    m=0;
    distance={$random} % 1024;
    end
    #500;
    knn_fin=1;
end

knn_result inst(
    .clk_en     (clk_en),
    .reset_n    (reset_n),
    .dic_go     (dic_go),

    .distance   (distance),
    .m          (m),

    .dic_end    (dic_end),
    .dic_end_q  (dic_end_q),
    .knn_fin    (knn_fin),

   
    .out_flag   (out_flag),
    .knn_resultf (knn_resultf),
    
    
    .min_o      (min_o),
    .min_p_o    (min_p_o),
    .max_p_o    (max_p_o),
    .max_o      (max_o),
    .max_cnt_o   (max_cnt_o),
    .q0_o       (q0_o),
    .q1_o       (q1_o),
    .fin_o      (fin_o),
    .stop_o     (stop_o),
    
    .color_cnt_0    (color_cnt_0),
    .color_cnt_1    (color_cnt_1),
    .color_cnt_2    (color_cnt_2),
    .color_cnt_3    (color_cnt_3),
    .color_cnt_4    (color_cnt_4)
   
);


endmodule