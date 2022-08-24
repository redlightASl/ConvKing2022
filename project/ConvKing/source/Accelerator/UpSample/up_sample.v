module up_sampling #(
           parameter data_i_width = 2,                  //输入数据矩阵长
           parameter data_i_height = 2,                 //输入数据矩阵宽

           parameter scale_factor = 2,                  //最近邻采样大小倍数

           parameter data_o_width = data_i_width * scale_factor,      //输出数据矩阵长
           parameter data_o_height = data_i_height * scale_factor,    //输出数据矩阵宽
           parameter bitwidth = 3
       ) (
           input [data_i_width * data_i_height * bitwidth - 1: 0] data_i,
           output [data_o_width * data_o_height * bitwidth - 1: 0] data_o
       );
//输入和对应的输出连起来
generate
    genvar i, j, m, n;
    for (i = 0;i < data_i_height;i = i + 1) begin
        for (j = 0;j < data_i_width;j = j + 1) begin
            for (m = i * scale_factor;m < i * scale_factor + scale_factor;m = m + 1) begin
                for (n = j * scale_factor;n < j * scale_factor + scale_factor;n = n + 1) begin
                    assign data_o [(m * data_o_width + n) * bitwidth + : bitwidth] = data_i [(i * data_i_width + j) * bitwidth + : bitwidth];
                end
            end
        end
    end
endgenerate


endmodule
