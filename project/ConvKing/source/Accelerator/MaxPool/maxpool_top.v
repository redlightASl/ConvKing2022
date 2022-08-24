module maxpool_top #(
    parameter datai_width = 4,                  //�������ݾ���
    parameter datai_height = 4,                 //�������ݾ����

    parameter kernel_width = 2,                 //���ط�Χ����
    parameter kernel_height = 2,                //���ط�Χ�����
    parameter stride = 2,                       //����

    parameter padding_en =0,                    //padding����
    parameter padding = 0,                      //padding������
  
    parameter datao_width = ((datai_width-kernel_width+2*padding)/stride)+1,        //�������ĳ�
    parameter datao_height = ((datai_height-kernel_height+2*padding)/stride)+1,     //�������Ŀ�

    parameter bitwidth = 3                      //λ��
) (
    input clk_en,                               //ʱ��
    input reset_n,                              //��λ
    
    input work_en,                              //����ʹ��
    input [datai_width*datai_height*bitwidth-1:0] data_i,       //��������
    
    output [datao_width*datao_height*bitwidth-1:0] data_o,      //�������
    output work_fin                                             //��������
);

reg state;                      //״̬��־
localparam IDLE = 0;            //����
localparam BUSY = 1;            //����

wire all_fin;                   //�ڲ��������ɹ�����־
wire pool_on;                   //�����ڲ��Ĺ�����־λ
reg rpool_on;
reg rpool_fin;
assign pool_on = rpool_on;
assign work_fin = rpool_fin;
always @(posedge clk_en ) begin
    if(!reset_n)begin
        state<=0;
        rpool_on<=0;
        rpool_fin<=0;
    end
    else begin
        case (state)
            IDLE:begin
                if(work_en)begin
                    state<=BUSY;
                    rpool_on <= 1;
                    rpool_fin <= 0;
                end
                else begin
                    state<=state;
                    rpool_fin<=rpool_fin;
                    rpool_on<=rpool_on;
                end
            end
            BUSY:begin
                if(all_fin)begin
                    state <= IDLE;
                    rpool_fin<= 1;
                    rpool_on<=0;
                end
                else begin
                    state <= state;
                    rpool_fin<=rpool_fin;
                    rpool_on<=rpool_on;
                end
            end
        endcase
    end
end

wire [3:0] data_l;                  //����������
wire [3:0] data_c;                  //����������
wire [3:0] resu_l;                  //���������
wire [3:0] resu_c;                  //���������

wire part_fin;                      //һ����Χ�������
wire turn_fin;                      //ȫ���������

wire [bitwidth-1:0] data;           //����������ݴ���

//����ģ��
maxpool_poolturn #(
    datai_width,
    datai_height,
    kernel_width,
    kernel_height,
    stride,
    padding_en,
    padding,
    datao_width,
    datao_height,
    bitwidth
)maxpool_poolturn_inst(
    .clk_en     (clk_en),
    .reset_n    (reset_n),
    .pool_on    (pool_on),

    .data_l     (data_l),
    .data_c     (data_c),
    .resu_l     (resu_l),
    .resu_c     (resu_c),

    .part_fin   (part_fin),
    .turn_fin   (turn_fin)
);

//������ȡģ��
maxpool_pick #(
    datai_width,
    datai_height,
    kernel_width,
    kernel_height,
    stride,
    padding_en,
    padding,
    datao_width,
    datao_height,
    bitwidth  
)maxpool_pick_inst(
    .clk_en     (clk_en),
    .reset_n    (reset_n),
    .pool_on    (pool_on),

    .data_i     (data_i),

    .data_l     (data_l),
    .data_c     (data_c),

    .data_o     (data)
);

//�Ƚ�ģ��
maxpool_compare #(
    datai_width,
    datai_height,
    kernel_width,
    kernel_height,
    stride,
    padding_en,
    padding,
    datao_width,
    datao_height,
    bitwidth  
)maxpool_compare_inst(
    .clk_en     (clk_en),
    .reset_n    (reset_n),
    .pool_on    (pool_on),
    .part_fin   (part_fin),
    .turn_fin   (turn_fin),

    .data       (data),
    .resu_l     (resu_l),
    .resu_c     (resu_c),

    .data_o     (data_o),
    .all_fin    (all_fin)
);


    
endmodule