module knn_top #(
           parameter knn = 2,                //中心点周围包括中心点像素数量，pixel_num=(knn*2)^2
           parameter color_num = 5          //训练集颜色数量
       ) (
           input clk_en,                 //系统时钟
           input reset_n,                //复位使能，低电平有效
           input knn_en,                 //knn使能

           input [9: 0] postion_lu_x,    //物体矩阵框左上角横坐标（列数）
           input [9: 0] postion_lu_y,    //物体矩阵框左上角纵坐标（行数）
           input [9: 0] postion_rd_x,    //物体矩阵框右下角...
           input [9: 0] postion_rd_y,    //物体矩阵框右下角...

           output [3: 0] knn_result,      //结果，输出为训练集颜色标号
           output knn_outflag     //输出标志位（高有效）

       );

wire dic_end;                      //训练集遍历完成标志位
wire dic_end_q;                    //训练集遍历完成标志位延时一个时钟周期
wire dic_go;                       //训练集遍历以及比较开始工作标志


wire [9: 0] i;                      //行遍历标志
wire [9: 0] j;                      //列遍历标志
wire [2: 0] m;                      //训练集遍历标志

reg [9: 0] ri;                      //行遍历标志寄存器
reg [9: 0] rj;                      //列遍历标志寄存器
reg [2: 0] rm;                      //训练集遍历标志

wire [13: 0] distance;              //欧式距离

reg [15: 0] img [0: 4][0: 4];       //图像二维数组
reg [15: 0] dic [0: color_num - 1];  //训练集一维数组
wire [15: 0] img_link;             //将像素信息送入距离计算模块
wire [15: 0] dic_link;             //将训练集信息送入距离计算模块
wire knn_fin;              //knn遍历完成标志

always@( * ) begin
    ri = i;
    rj = j;
    rm = m;
end

always@( * ) begin
    if (!knn_en) begin                //数组初始化（测试用）
        img[0][0] = 16'h0000;
        img[0][1] = 16'h001F;
        img[0][2] = 16'h001F;
        img[0][3] = 16'hFFE0;
        img[0][4] = 16'hFFE0;
        img[1][0] = 16'hFFE0;
        img[1][1] = 16'hFFE0;
        img[1][2] = 16'h7800;
        img[1][3] = 16'h0000;
        img[1][4] = 16'hFFE0;
        img[2][0] = 16'hFFE0;
        img[2][1] = 16'hFFE0;
        img[2][2] = 16'hFFE0;
        img[2][3] = 16'hFFE0;
        img[2][4] = 16'hFFE0;
        img[3][0] = 16'h0000;
        img[3][1] = 16'hFFE0;
        img[3][2] = 16'hFFE0;
        img[3][3] = 16'hFFE0;
        img[3][4] = 16'hFFE0;
        img[4][0] = 16'hFFE0;
        img[4][1] = 16'hFFE0;
        img[4][2] = 16'hFFE0;
        img[4][3] = 16'hFFE0;
        img[4][4] = 16'h001F;

        dic[0] = 16'h7800;        //深红色
        dic[1] = 16'hFFE0;        //黄色
        dic[2] = 16'h001F;        //蓝色
        dic[3] = 16'h0000;        //黑色
        dic[4] = 16'hFFFF;        //白色
    end
end
assign img_link = img[ri][rj];
assign dic_link = dic[rm];

//图片数组位置遍历模块
knn_img # (
            knn
        )knn_img_inst(
            .clk_en (clk_en),
            .reset_n (reset_n),

            .dic_end (dic_end),
            .dic_end_q (dic_end_q),
            .knn_en (knn_en),

            .postion_lu_x (postion_lu_x),
            .postion_lu_y (postion_lu_y),
            .postion_rd_x (postion_rd_x),
            .postion_rd_y (postion_rd_y),

            .i (i),
            .j (j),

            .knn_fin_o (knn_fin),
            .dic_go_o (dic_go)
        );

//数据集位置遍历模块
knn_dic # (
            color_num
        )knn_dic_inst(
            .clk_en (clk_en),
            .reset_n (reset_n),

            .knn_fin (knn_fin),
            .dic_go (dic_go),
            .dic_end (dic_end),
            .dic_end_q_o (dic_end_q),
            .m (m)
        );

//欧式距离计算模块
knn_distance knn_distance_inst(
                 .img (img_link),
                 .dic (dic_link),

                 .distance (distance)
             );

//数据处理及结果输出模块
knn_result # (
               knn,
               color_num
           )knn_result_inst(
               .clk_en (clk_en),
               .reset_n (reset_n),
               .dic_go (dic_go),

               .distance (distance),
               .m (m),
               .dic_end (dic_end),
               .dic_end_q (dic_end_q),
               .knn_fin (knn_fin),

               .knn_resultf (knn_result),
               .out_flag (knn_outflag)
           );

endmodule
