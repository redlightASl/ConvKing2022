module knn_result #(
           parameter knn = 4,
           parameter color_num = 5
       ) (
           input clk_en,
           input reset_n,
           input dic_go,                //训练集遍历以及比较开始工作标志

           input [13: 0] distance,    //欧式距离
           input [2: 0] m,           //训练集遍历标志

           input dic_end,               //训练集遍历完成标志位
           input dic_end_q,             //训练集遍历完成标志位延时一个时钟周期
           input knn_fin,               //knn遍历完成标志

           output [3: 0] knn_resultf,    //结果，输出为训练集颜色标号
           output out_flag             //输出标志位（高有效）
       );

//距离最小值寻找块，在数据集移位并且距离算出后马上比较
reg [13: 0] min;       //暂存最小值
reg [2: 0] min_p;     //储存最小值对应的颜色标号
always @(posedge clk_en) begin
    if (!reset_n) begin
        min <= 0;
        min_p <= 0;
    end
    else if (dic_go) begin   //和训练集遍历进行流水线交替工作
        if (m == 0) begin      //第一个数据集算出的距离??
            min <= distance;
            min_p <= m;
        end
        else if (m > 0 & dic_end & dic_end_q) begin     //在数据集遍历结束后保持给计数统计留有时间
            min <= min;
            min_p <= min_p;
        end
        else if (m > 0 & (!(dic_end & dic_end_q))) begin   //比较
            if (min > distance) begin
                min <= distance;
                min_p <= m;
            end
            else begin
                min <= min;
                min_p <= min_p;
            end
        end
        else begin
            min <= min;
            min_p <= min_p;
        end
    end
    else begin
        min <= 0;
        min_p <= 0;
    end
end

/*knn概率算法*/
//即哪个颜色在knn遍历块中占比高，即该块为哪个颜色
//在判断出一个像素块对应的颜色标号后，相应颜色对应的矩阵+1统计
reg [5: 0] color_cnt [0: color_num - 1];      //统计各颜色所占的像素数量
always@(posedge clk_en) begin
    if (!reset_n) begin
        color_cnt[0] <= 0;
        color_cnt[1] <= 0;
        color_cnt[2] <= 0;
        color_cnt[3] <= 0;
        color_cnt[4] <= 0;
    end
    else if (dic_end & dic_end_q) begin    //工作在数据集遍历完成后的一个时钟周期
        color_cnt[min_p] <= color_cnt[min_p] + 1;
    end
    else begin
        color_cnt[0] <= color_cnt[0];
        color_cnt[1] <= color_cnt[1];
        color_cnt[2] <= color_cnt[2];
        color_cnt[3] <= color_cnt[3];
        color_cnt[4] <= color_cnt[4];
    end
end

//knn遍历完成标志上升沿以进行初始化
reg q0;
reg q1;
wire fin;
always@(posedge clk_en) begin
    if (!reset_n) begin
        q0 <= 0;
    end
    else begin
        q0 <= knn_fin;
        q1 <= q0;
    end
end
assign fin = ~q1 & q0;

reg [3: 0] max_cnt;  //统计矩阵遍历计数
reg [5: 0] max;      //最大值暂存
reg [3: 0] max_p;    //最大值对应的颜色标号
wire stop;          //遍历结束标志
assign stop = (max_cnt == color_num) ? 1 : 0;
always@(posedge clk_en) begin
    if (!reset_n) begin
        max <= 0;
        max_p <= 0;
        max_cnt <= 0;
    end
    else if (fin) begin               //初始化
        max_cnt <= max_cnt + 1;
        max_p <= 0;
        max <= color_cnt[max_cnt];
    end
    else if (q1 & q0 & !stop) begin               //knn遍历结束之后找寻统计矩阵中的最大值所对应的颜色标号
        if (color_cnt[max_cnt] > max) begin
            max <= color_cnt[max_cnt];
            max_p <= max_cnt;
        end
        else begin
            max_p <= max_p;
            max <= max;
        end
        max_cnt <= max_cnt + 1;
    end
    else begin
        max_cnt <= max_cnt;
    end
end
assign knn_resultf = stop ? max_p : 0;           //遍历完成后将knn块像素中占比最多的颜色标号输出，即结果
assign out_flag = stop ? 1 : 0;                  //结果输出标志位

endmodule
