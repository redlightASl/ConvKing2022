module maxpool_compare #(
    parameter datai_width = 4,
    parameter datai_height = 4,

    parameter kernel_width = 2,
    parameter kernel_height = 2,
    parameter stride = 2,

    parameter padding_en =0,
    parameter padding = 0,

    parameter datao_width = ((datai_width-kernel_width+2*padding)/stride)+1,
    parameter datao_height = ((datai_height-kernel_height+2*padding)/stride)+1,

    parameter bitwidth = 3 
) (
    input clk_en,
    input reset_n,
    input pool_on,
    input part_fin,
    input turn_fin,  

    input [bitwidth-1:0] data,
    input [3:0] resu_l,
    input [3:0] resu_c,

    output [datao_width*datao_height*bitwidth-1:0] data_o,
    output all_fin
);

reg part_fin_q;             //延时一拍给比较操作完成
always @(posedge clk_en) begin
    if(!reset_n)begin
        part_fin_q<=0;
    end
    else begin
        part_fin_q<=part_fin;
    end
end

//比最大值，并将其输入到结果矩阵
reg [3:0] i;
reg [3:0] j;
reg [bitwidth-1:0] result_array [0:datao_height-1][0:datao_width-1];
always @(posedge clk_en) begin
    if(!reset_n)begin
        for(i=0;i<datao_height;i=i+1)begin
            for(j=0;j<datao_width;j=j+1)begin
                result_array[i][j]<=0;
            end
        end
    end
    else if (pool_on) begin
        if (turn_fin) begin
            for(i=0;i<datao_height;i=i+1)begin
                for(j=0;j<datao_width;j=j+1)begin
                    result_array[i][j]<=result_array[i][j];
                end
            end
        end
        else begin
            if(part_fin_q)begin
                result_array[resu_l][resu_c]<=data;
            end
            else begin
                if(result_array[resu_l][resu_c]<data)begin
                    result_array[resu_l][resu_c]<=data;
                end
                else begin
                    result_array[resu_l][resu_c]<=result_array[resu_l][resu_c];
                end
            end
        end
    end
    else begin
        for(i=0;i<datao_height;i=i+1)begin
            for(j=0;j<datao_width;j=j+1)begin
                result_array[i][j]<=0;
            end
        end
    end
end

//将结果矩阵的输出数据转成串行输出
reg [3:0] x;
reg [3:0] y;
reg [datao_width*datao_height*bitwidth-1:0] rdata_o;
assign data_o = rdata_o;
always @(posedge clk_en)begin
    if(!reset_n)begin
        rdata_o<=0;
    end
    else if(turn_fin)begin
        for(x=0;x<datao_height;x=x+1)begin
            for(y=0;y<datao_width;y=y+1)begin
                rdata_o[(x*datao_width+y)*bitwidth +:bitwidth] <= result_array[x][y]; 
            end
        end
    end
    else if(all_fin)begin
        rdata_o<=rdata_o;
    end
    else begin
        rdata_o<=0;
    end
end

//返回工作完成标志给状态机变换状态
reg rall_fin;
assign all_fin=rall_fin;
always @(*)begin
    if(!reset_n)begin
        rall_fin<=0;
    end
    else if(turn_fin)begin
        rall_fin<=1;
    end
    else if(pool_on)begin
        rall_fin<=0;
    end
    else begin
        rall_fin<=rall_fin;
    end
end

endmodule