module knn_result #(
           parameter knn = 4,
           parameter color_num = 5
       ) (
           input clk_en,
           input reset_n,
           input dic_go,                //ѵ���������Լ��ȽϿ�ʼ������־

           input [13: 0] distance,    //ŷʽ����
           input [2: 0] m,           //ѵ����������־

           input dic_end,               //ѵ����������ɱ�־λ
           input dic_end_q,             //ѵ����������ɱ�־λ��ʱһ��ʱ������
           input knn_fin,               //knn������ɱ�־

           output [3: 0] knn_resultf,    //��������Ϊѵ������ɫ���
           output out_flag             //�����־λ������Ч��
       );

//������СֵѰ�ҿ飬�����ݼ���λ���Ҿ�����������ϱȽ�
reg [13: 0] min;       //�ݴ���Сֵ
reg [2: 0] min_p;     //������Сֵ��Ӧ����ɫ���
always @(posedge clk_en) begin
    if (!reset_n) begin
        min <= 0;
        min_p <= 0;
    end
    else if (dic_go) begin   //��ѵ��������������ˮ�߽��湤��
        if (m == 0) begin      //��һ�����ݼ�����ľ���??
            min <= distance;
            min_p <= m;
        end
        else if (m > 0 & dic_end & dic_end_q) begin     //�����ݼ����������󱣳ָ�����ͳ������ʱ��
            min <= min;
            min_p <= min_p;
        end
        else if (m > 0 & (!(dic_end & dic_end_q))) begin   //�Ƚ�
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

/*knn�����㷨*/
//���ĸ���ɫ��knn��������ռ�ȸߣ����ÿ�Ϊ�ĸ���ɫ
//���жϳ�һ�����ؿ��Ӧ����ɫ��ź���Ӧ��ɫ��Ӧ�ľ���+1ͳ��
reg [5: 0] color_cnt [0: color_num - 1];      //ͳ�Ƹ���ɫ��ռ����������
always@(posedge clk_en) begin
    if (!reset_n) begin
        color_cnt[0] <= 0;
        color_cnt[1] <= 0;
        color_cnt[2] <= 0;
        color_cnt[3] <= 0;
        color_cnt[4] <= 0;
    end
    else if (dic_end & dic_end_q) begin    //���������ݼ�������ɺ��һ��ʱ������
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

//knn������ɱ�־�������Խ��г�ʼ��
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

reg [3: 0] max_cnt;  //ͳ�ƾ����������
reg [5: 0] max;      //���ֵ�ݴ�
reg [3: 0] max_p;    //���ֵ��Ӧ����ɫ���
wire stop;          //����������־
assign stop = (max_cnt == color_num) ? 1 : 0;
always@(posedge clk_en) begin
    if (!reset_n) begin
        max <= 0;
        max_p <= 0;
        max_cnt <= 0;
    end
    else if (fin) begin               //��ʼ��
        max_cnt <= max_cnt + 1;
        max_p <= 0;
        max <= color_cnt[max_cnt];
    end
    else if (q1 & q0 & !stop) begin               //knn��������֮����Ѱͳ�ƾ����е����ֵ����Ӧ����ɫ���
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
assign knn_resultf = stop ? max_p : 0;           //������ɺ�knn��������ռ��������ɫ�������������
assign out_flag = stop ? 1 : 0;                  //��������־λ

endmodule
