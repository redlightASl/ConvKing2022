module knn_top #(
           parameter knn = 2,                //���ĵ���Χ�������ĵ�����������pixel_num=(knn*2)^2
           parameter color_num = 5          //ѵ������ɫ����
       ) (
           input clk_en,                 //ϵͳʱ��
           input reset_n,                //��λʹ�ܣ��͵�ƽ��Ч
           input knn_en,                 //knnʹ��

           input [9: 0] postion_lu_x,    //�����������ϽǺ����꣨������
           input [9: 0] postion_lu_y,    //�����������Ͻ������꣨������
           input [9: 0] postion_rd_x,    //�����������½�...
           input [9: 0] postion_rd_y,    //�����������½�...

           output [3: 0] knn_result,      //��������Ϊѵ������ɫ���
           output knn_outflag     //�����־λ������Ч��

       );

wire dic_end;                      //ѵ����������ɱ�־λ
wire dic_end_q;                    //ѵ����������ɱ�־λ��ʱһ��ʱ������
wire dic_go;                       //ѵ���������Լ��ȽϿ�ʼ������־


wire [9: 0] i;                      //�б�����־
wire [9: 0] j;                      //�б�����־
wire [2: 0] m;                      //ѵ����������־

reg [9: 0] ri;                      //�б�����־�Ĵ���
reg [9: 0] rj;                      //�б�����־�Ĵ���
reg [2: 0] rm;                      //ѵ����������־

wire [13: 0] distance;              //ŷʽ����

reg [15: 0] img [0: 4][0: 4];       //ͼ���ά����
reg [15: 0] dic [0: color_num - 1];  //ѵ����һά����
wire [15: 0] img_link;             //��������Ϣ����������ģ��
wire [15: 0] dic_link;             //��ѵ������Ϣ����������ģ��
wire knn_fin;              //knn������ɱ�־

always@( * ) begin
    ri = i;
    rj = j;
    rm = m;
end

always@( * ) begin
    if (!knn_en) begin                //�����ʼ���������ã�
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

        dic[0] = 16'h7800;        //���ɫ
        dic[1] = 16'hFFE0;        //��ɫ
        dic[2] = 16'h001F;        //��ɫ
        dic[3] = 16'h0000;        //��ɫ
        dic[4] = 16'hFFFF;        //��ɫ
    end
end
assign img_link = img[ri][rj];
assign dic_link = dic[rm];

//ͼƬ����λ�ñ���ģ��
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

//���ݼ�λ�ñ���ģ��
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

//ŷʽ�������ģ��
knn_distance knn_distance_inst(
                 .img (img_link),
                 .dic (dic_link),

                 .distance (distance)
             );

//���ݴ���������ģ��
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
