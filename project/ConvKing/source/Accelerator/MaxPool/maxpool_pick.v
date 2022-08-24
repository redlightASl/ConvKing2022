module maxpool_pick #(
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
    
    input [datai_width*datai_height*bitwidth-1:0] data_i,           //输入数据

    input [3:0] data_l,                     //输入数据行
    input [3:0] data_c,                     //输入数据列

    output [bitwidth-1:0] data_o            //提取出的输入矩阵数据
);

//矩阵展开
wire [bitwidth-1:0] data [0:datai_height+2*padding-1][0:datai_width+2*padding-1];       //数据数据矩阵
generate
    genvar i,j;  
    for(i=0;i<datai_height+2*padding;i=i+1)begin
        for (j=0;j<datai_width+2*padding;j=j+1)begin
            if (padding_en) begin       //padding处理块
                if (i<padding|i>datai_height|j<padding|j>datai_width) begin
                    assign data[i][j] = 0;
                end
                else begin
                    assign data[i][j] = data_i[((i-padding)*datai_width+j-padding)*bitwidth +:bitwidth];
                end       
            end
            else begin
                assign data[i][j] = data_i[(i*datai_width+j)*bitwidth +:bitwidth];
            end
        end
    end
endgenerate

//将相应的输入数据提出
reg [3:0] rdata_l;
reg [3:0] rdata_c;
assign data_o = data[rdata_l][rdata_c];
always @(*)begin
    if(pool_on)begin
        rdata_l=data_l;
        rdata_c=data_c;
    end
    else begin
        rdata_l=0;
        rdata_c=0;
    end
end

endmodule