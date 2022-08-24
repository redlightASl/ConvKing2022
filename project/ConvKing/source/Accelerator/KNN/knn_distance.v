module knn_distance (
           input [15: 0] img,             //图片像素RGB56信息
           input [15: 0] dic,             //训练集RGB565信息

           output [13: 0] distance        //欧式距离
       );

wire [4: 0] color_R_img;         //图片像素RGB565拆分值
wire [5: 0] color_G_img;
wire [4: 0] color_B_img;
wire [4: 0] color_R_dic;         //数据集RGB565拆分值
wire [5: 0] color_G_dic;
wire [4: 0] color_B_dic;
assign color_R_img = img[15: 11];   //传递，计算，不开方（开根号不好搭电路）
assign color_G_img = img[10: 5];
assign color_B_img = img[4: 0];
assign color_R_dic = dic[15: 11];
assign color_G_dic = dic[10: 5];
assign color_B_dic = dic[4: 0];
assign distance = (color_R_img - color_R_dic) * (color_R_img - color_R_dic) + (color_G_img - color_G_dic) * (color_G_img - color_G_dic) + (color_B_img - color_B_dic) * (color_B_img - color_B_dic); //娆у紡璺濈?讳笉寮�鏂?

endmodule
