module u_sampling_top #(
           parameter data_i_width = 2,                  //输入数据矩阵长
           parameter data_i_height = 2,                 //输入数据矩阵宽

           parameter scale_factor = 2,                  //最近邻采样大小倍数

           parameter data_o_width = data_i_width * scale_factor,      //输出数据矩阵长
           parameter data_o_height = data_i_height * scale_factor,    //输出数据矩阵宽
           parameter bitwidth = 3              //位宽
       ) (
           input [data_i_width * data_i_height * bitwidth - 1: 0] data_i,
           output [data_o_width * data_o_height * bitwidth - 1: 0] data_o
       );

up_sampling #(
                data_i_width,
                data_i_height,
                scale_factor,
                data_o_width,
                data_o_height,
                bitwidth
            )
            up_sampling_inst(
                .data_i (data_i),
                .data_o (data_o)
            );

endmodule
