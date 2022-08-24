module knn_img #(
           parameter knn = 4
       ) (
           input clk_en,
           input reset_n,

           input dic_end,                  //训练集遍历完成标志位
           input dic_end_q,                //训练集遍历完成标志位延时一个时钟周期
           input knn_en,                   //knn使能

           input [9: 0] postion_lu_x,
           input [9: 0] postion_lu_y,
           input [9: 0] postion_rd_x,
           input [9: 0] postion_rd_y,

           output reg [9: 0] i,          //行遍历标志
           output reg [9: 0] j,          //列遍历标志

           output knn_fin_o,              //knn遍历完成标志
           output dic_go_o               //训练集遍历以及比较开始工作标志
       );

assign knn_fin_o = knn_fin;

//捕捉knn使能上升沿作初始化
reg q0;
reg q1;
wire knn_init;
always@(posedge clk_en) begin
    if (!reset_n) begin
        q0 <= 0;
        q1 <= 0;
    end
    else begin
        q0 <= knn_en;
        q1 <= q0;
    end
end
assign knn_init = ~q1 & q0;



//图片数组遍历块
reg [3: 0] cnt_w; //横坐标遍历计数(列）
reg [3: 0] cnt_h; //纵坐标遍历计数（行）
wire [9: 0] wid_center; //中点横坐标（列）
wire [9: 0] hei_center; //中点纵坐标（行）
wire cnt_w_en; //横坐标遍历计数完成标志
wire cnt_h_en; //纵坐标遍历计数完成标志
wire knn_fin; //knn矩阵遍历完成
reg dic_go; //训练集遍历开始工作
assign wid_center = (postion_lu_x + postion_rd_x) >> 1;   //右移一位表示除2
assign hei_center = (postion_lu_y + postion_rd_y) >> 1;
assign cnt_w_en = (cnt_w == (knn * 2 - 1)) ? 1 : 0;
assign cnt_h_en = (cnt_h == knn * 2) ? 1 : 0;
assign knn_fin = cnt_h_en ? 1 : 0;
assign dic_go_o = dic_go;
always @(posedge clk_en) begin
    if (!reset_n) begin
        i <= 0;
        j <= 0;
        cnt_w <= 0;
        cnt_h <= 0;
    end
    else if (knn_init) begin
        j <= wid_center - knn;          //遍历初始位置为左上角
        i <= hei_center - knn;
        cnt_w <= 0;
        cnt_h <= 0;
        dic_go <= 1;                  //训练集遍历开始工作
    end
    else if (!knn_fin & (dic_end & !dic_end_q)) begin    //训练集遍历完成
        if (cnt_w_en) begin         //一行遍历完成
            cnt_w <= 0;
            i <= i + 1;
            j <= hei_center - knn;
            cnt_h <= cnt_h + 1;
        end
        else begin
            j <= j + 1;
            cnt_w <= cnt_w + 1;
        end
    end
    else if (knn_fin) begin
        dic_go <= 0;
        i <= i;
        j <= j;
    end
    else begin
        i <= i;
        j <= j;
    end
end

endmodule
