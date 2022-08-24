module maxpool_poolturn #(
    parameter datai_width = 4,
    parameter datai_height = 4,

    parameter kernel_width = 2,
    parameter kernel_height = 2,
    parameter stride = 1,

    parameter padding_en =0,
    parameter padding = 0,

    parameter datao_width = ((datai_width-kernel_width+2*padding)/stride)+1,
    parameter datao_height = ((datai_height-kernel_height+2*padding)/stride)+1,

    parameter bitwidth = 3 
) (
    input clk_en,
    input reset_n,
    
    input pool_on,

    output [3:0] data_l,        //数据行
    output [3:0] data_c,        //数据列
    
    output [3:0] resu_l,        //结果矩阵行
    output [3:0] resu_c,        //结果矩阵列

    output part_fin,            //一个范围完成
    output turn_fin             //全部遍历完成
);



reg [3:0] rresu_l;
reg [3:0] rresu_c;
wire change_line;               //换行标志
assign change_line = (rresu_c == datao_width-1)?1:0;
assign turn_fin = (rresu_l==datao_height)?1:0;
assign resu_l = rresu_l;
assign resu_c = rresu_c;
always @(posedge clk_en) begin
    if(!reset_n)begin
        rresu_l<=0;
        rresu_c<=0;
    end
    else if (pool_on) begin
        if(turn_fin)begin
            rresu_l<=rresu_l;
            rresu_c<=rresu_c;
        end
        else begin
            if(part_fin&change_line)begin
                rresu_l<=rresu_l+1;
                rresu_c<=0;
            end
            else if(part_fin&~change_line)begin
                rresu_l<=rresu_l;
                rresu_c<=rresu_c+1;
            end
            else begin
                rresu_l<=rresu_l;
                rresu_c<=rresu_c;
            end
        end
    end
    else begin
        rresu_l<=0;
        rresu_c<=0;
    end
end

reg [3:0] datal_cnt;
reg [3:0] datac_cnt;
wire datal_fin;
wire datac_fin;
assign datac_fin = (datac_cnt==kernel_width-1)?1:0;                                     //遍历到一个框的最右端
assign part_fin = (datac_cnt==kernel_width-1&datal_cnt==kernel_height-1)?1:0;           //一个范围的遍历完成标志
assign resu_pos_change = (datac_cnt==kernel_width-2&datal_cnt==kernel_height-1)?1:0;    //锚点更换标志位
always @(posedge clk_en) begin
    if(!reset_n)begin
        datal_cnt<=0;
        datac_cnt<=0;
    end
    else if (pool_on) begin
        if(turn_fin)begin
            datal_cnt<=datal_cnt;
            datac_cnt<=datac_cnt;
        end
        else begin
            if(part_fin&datac_fin)begin
                datal_cnt<=0;
                datac_cnt<=0;
            end
            else if (~part_fin&datac_fin) begin
                datac_cnt<=0;
                datal_cnt<=datal_cnt+1;
            end
            else begin
                datac_cnt<=datac_cnt+1;
                datal_cnt<=datal_cnt;
            end
        end
    end
    else begin
        datac_cnt<=0;
        datal_cnt<=0;
    end
end

wire [3:0] anchor_l;            //锚点行
wire [3:0] anchor_c;            //锚点列
reg [3:0] ranchor_l;
reg [3:0] ranchor_c;
assign anchor_l=ranchor_l;
assign anchor_c=ranchor_c;
always @(*)begin
    if(pool_on)begin
        if(turn_fin)begin
            ranchor_l=ranchor_l;
            ranchor_c=ranchor_c;
        end
        else begin
            if(resu_pos_change&change_line)begin
                ranchor_l=ranchor_l+stride;
                ranchor_c=0;
            end
            else if(resu_pos_change&~change_line)begin
                ranchor_l=ranchor_l;
                ranchor_c=ranchor_c+stride;
            end
            else begin
                ranchor_l=ranchor_l;
                ranchor_c=ranchor_c;
            end
        end
    end
    else begin
        ranchor_l=0;
        ranchor_c=0;
    end
end

reg [3:0] rdata_l;              //输入数据行
reg [3:0] rdata_c;              //输入数据列
assign data_l = rdata_l;        
assign data_c = rdata_c;
always @(posedge clk_en)begin
    if(!reset_n)begin
        rdata_l<=0;
        rdata_c<=0;
    end
    else if (pool_on) begin
        if(turn_fin)begin
            rdata_l<=rdata_l;
            rdata_c<=rdata_c;
        end
        else begin
            if(part_fin&datac_fin)begin
                rdata_l<=anchor_l;
                rdata_c<=anchor_c;
            end
            else if(~part_fin&datac_fin) begin
                rdata_l<=rdata_l+1;
                rdata_c<=anchor_c;
            end
            else begin
                rdata_l<=rdata_l;
                rdata_c<=rdata_c+1;
            end
        end
    end
    else begin
        rdata_l<=0;
        rdata_c<=0;
    end
end

endmodule