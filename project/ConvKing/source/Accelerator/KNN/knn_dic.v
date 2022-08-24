module knn_dic #(
           parameter color_num = 5
       ) (
           input clk_en,
           input reset_n,
           input dic_go,            //训练集遍历以及比较开始工作标志
           input knn_fin,           //knn遍历完成标志

           output reg [2: 0] m,      //训练集遍历标志
           output dic_end,         //训练集遍历完成标志位
           output dic_end_q_o     //训练集遍历完成标志位延时一个时钟周期
       );

//延时一时钟周期模块
reg dic_end_q;
always@(posedge clk_en) begin
    if (!reset_n) begin
        dic_end_q <= 0;
    end
    else begin
        dic_end_q <= dic_end;
    end
end
assign dic_end_q_o = dic_end_q;


//训练集数组遍历块
reg [3: 0] cnt_dic;                             //训练集遍历计数
assign dic_end = (cnt_dic == color_num - 1) ? 1 : 0;     //训练集遍历完成
always @(posedge clk_en) begin
    if (!reset_n) begin
        cnt_dic <= 0;
        m <= 0;
    end
    else if (dic_go) begin
        if (dic_end_q & dic_end) begin      //遍历完成后清零
            m <= 0;
            cnt_dic <= 0;
        end
        else if (dic_end) begin           //保持给距离比较留有一个时钟周期
            m <= m;
            cnt_dic <= cnt_dic;
        end
        else begin                        //遍历计数啦
            m <= m + 1;
            cnt_dic <= cnt_dic + 1;
        end
    end
    else if (knn_fin & !dic_go) begin         //knn结束之后也得保持
        cnt_dic <= cnt_dic;
        m <= m;
    end
    else begin
        cnt_dic <= 0;
        m <= 0;
    end
end

endmodule
