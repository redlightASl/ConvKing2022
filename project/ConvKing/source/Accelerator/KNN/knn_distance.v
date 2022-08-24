module knn_distance (
           input [15: 0] img,             //ͼƬ����RGB56��Ϣ
           input [15: 0] dic,             //ѵ����RGB565��Ϣ

           output [13: 0] distance        //ŷʽ����
       );

wire [4: 0] color_R_img;         //ͼƬ����RGB565���ֵ
wire [5: 0] color_G_img;
wire [4: 0] color_B_img;
wire [4: 0] color_R_dic;         //���ݼ�RGB565���ֵ
wire [5: 0] color_G_dic;
wire [4: 0] color_B_dic;
assign color_R_img = img[15: 11];   //���ݣ����㣬�������������Ų��ô��·��
assign color_G_img = img[10: 5];
assign color_B_img = img[4: 0];
assign color_R_dic = dic[15: 11];
assign color_G_dic = dic[10: 5];
assign color_B_dic = dic[4: 0];
assign distance = (color_R_img - color_R_dic) * (color_R_img - color_R_dic) + (color_G_img - color_G_dic) * (color_G_img - color_G_dic) + (color_B_img - color_B_dic) * (color_B_img - color_B_dic); //欧式距�?�不开�?

endmodule
