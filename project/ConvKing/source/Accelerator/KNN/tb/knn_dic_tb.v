`timescale 1ns/1ps
module  knn_dic_tb();

reg clk_en;
reg reset_n;
reg dic_go;

wire dic_end_q_o;

wire [2:0]  m_o;
wire [2:0]  cnt_dic_o;
wire dic_end;

initial begin
    clk_en=0;
    forever #100 clk_en=~clk_en;
end

initial begin
    dic_go=0;
    #500    dic_go=1;
end

initial begin
    reset_n=0;
    #400 reset_n=1;
    #8000 reset_n=0;
end




knn_dic inst(
    .clk_en     (clk_en),
    .reset_n    (reset_n),
    
    .dic_go     (dic_go),
    .dic_end_q_o(dic_end_q_o),
    
    .m_o        (m_o),
    .cnt_dic_o  (cnt_dic_o),
    .dic_end    (dic_end)
);



endmodule

