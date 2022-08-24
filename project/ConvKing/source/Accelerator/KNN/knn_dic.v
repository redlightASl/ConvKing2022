module knn_dic #(
           parameter color_num = 5
       ) (
           input clk_en,
           input reset_n,
           input dic_go,            //ѵ���������Լ��ȽϿ�ʼ������־
           input knn_fin,           //knn������ɱ�־

           output reg [2: 0] m,      //ѵ����������־
           output dic_end,         //ѵ����������ɱ�־λ
           output dic_end_q_o     //ѵ����������ɱ�־λ��ʱһ��ʱ������
       );

//��ʱһʱ������ģ��
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


//ѵ�������������
reg [3: 0] cnt_dic;                             //ѵ������������
assign dic_end = (cnt_dic == color_num - 1) ? 1 : 0;     //ѵ�����������
always @(posedge clk_en) begin
    if (!reset_n) begin
        cnt_dic <= 0;
        m <= 0;
    end
    else if (dic_go) begin
        if (dic_end_q & dic_end) begin      //������ɺ�����
            m <= 0;
            cnt_dic <= 0;
        end
        else if (dic_end) begin           //���ָ�����Ƚ�����һ��ʱ������
            m <= m;
            cnt_dic <= cnt_dic;
        end
        else begin                        //����������
            m <= m + 1;
            cnt_dic <= cnt_dic + 1;
        end
    end
    else if (knn_fin & !dic_go) begin         //knn����֮��Ҳ�ñ���
        cnt_dic <= cnt_dic;
        m <= m;
    end
    else begin
        cnt_dic <= 0;
        m <= 0;
    end
end

endmodule
