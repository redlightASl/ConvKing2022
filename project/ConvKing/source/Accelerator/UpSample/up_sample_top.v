module u_sampling_top #(
           parameter data_i_width = 2,                  //�������ݾ���
           parameter data_i_height = 2,                 //�������ݾ����

           parameter scale_factor = 2,                  //����ڲ�����С����

           parameter data_o_width = data_i_width * scale_factor,      //������ݾ���
           parameter data_o_height = data_i_height * scale_factor,    //������ݾ����
           parameter bitwidth = 3              //λ��
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
