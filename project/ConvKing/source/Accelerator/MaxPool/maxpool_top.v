module maxpool_top #(
    parameter datai_width = 4,                  //输入数据矩阵长
    parameter datai_height = 4,                 //输入数据矩阵宽

    parameter kernel_width = 2,                 //最大池范围矩阵长
    parameter kernel_height = 2,                //最大池范围矩阵宽
    parameter stride = 2,                       //步长

    parameter padding_en =0,                    //padding开关
    parameter padding = 0,                      //padding行列数
  
    parameter datao_width = ((datai_width-kernel_width+2*padding)/stride)+1,        //输出矩阵的长
    parameter datao_height = ((datai_height-kernel_height+2*padding)/stride)+1,     //输出矩阵的宽

    parameter bitwidth = 3                      //位宽
) (
    input clk_en,                               //时钟
    input reset_n,                              //复位
    
    input work_en,                              //工作使能
    input [datai_width*datai_height*bitwidth-1:0] data_i,       //数据输入
    
    output [datao_width*datao_height*bitwidth-1:0] data_o,      //数据输出
    output work_fin                                             //工作结束
);

reg state;                      //状态标志
localparam IDLE = 0;            //空闲
localparam BUSY = 1;            //工作

wire all_fin;                   //内部输出的完成工作标志
wire pool_on;                   //输入内部的工作标志位
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

wire [3:0] data_l;                  //输入数据行
wire [3:0] data_c;                  //输入数据列
wire [3:0] resu_l;                  //结果矩阵行
wire [3:0] resu_c;                  //结果矩阵列

wire part_fin;                      //一个范围遍历完成
wire turn_fin;                      //全部遍历完成

wire [bitwidth-1:0] data;           //输入矩阵数据传递

//遍历模块
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

//数据提取模块
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

//比较模块
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