module knn_img #(
           parameter knn = 4
       ) (
           input clk_en,
           input reset_n,

           input dic_end,                  //ѵ����������ɱ�־λ
           input dic_end_q,                //ѵ����������ɱ�־λ��ʱһ��ʱ������
           input knn_en,                   //knnʹ��

           input [9: 0] postion_lu_x,
           input [9: 0] postion_lu_y,
           input [9: 0] postion_rd_x,
           input [9: 0] postion_rd_y,

           output reg [9: 0] i,          //�б�����־
           output reg [9: 0] j,          //�б�����־

           output knn_fin_o,              //knn������ɱ�־
           output dic_go_o               //ѵ���������Լ��ȽϿ�ʼ������־
       );

assign knn_fin_o = knn_fin;

//��׽knnʹ������������ʼ��
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



//ͼƬ���������
reg [3: 0] cnt_w; //�������������(�У�
reg [3: 0] cnt_h; //����������������У�
wire [9: 0] wid_center; //�е�����꣨�У�
wire [9: 0] hei_center; //�е������꣨�У�
wire cnt_w_en; //���������������ɱ�־
wire cnt_h_en; //���������������ɱ�־
wire knn_fin; //knn����������
reg dic_go; //ѵ����������ʼ����
assign wid_center = (postion_lu_x + postion_rd_x) >> 1;   //����һλ��ʾ��2
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
        j <= wid_center - knn;          //������ʼλ��Ϊ���Ͻ�
        i <= hei_center - knn;
        cnt_w <= 0;
        cnt_h <= 0;
        dic_go <= 1;                  //ѵ����������ʼ����
    end
    else if (!knn_fin & (dic_end & !dic_end_q)) begin    //ѵ�����������
        if (cnt_w_en) begin         //һ�б������
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
