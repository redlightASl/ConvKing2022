module conv_buffer #(
    parameter weight_width = 2,         
    parameter weight_height = 2,       

    parameter img_width = 4,           
    parameter img_height = 4,          
    
    parameter padding_enable = 0,     
    parameter padding = 0,             

    parameter stride = 1,               
    parameter bitwidth = 3,            
    parameter result_width = (img_width-weight_width+2*padding)/stride+1,      
    parameter result_height = (img_height-weight_height+2*padding)/stride+1,     
    parameter expand = 1        
) (
    input clk_en,
    input rst_n,
    input conv_on,
    
    input [31:0] anchor_l,
    input [31:0] anchor_c,

    input [3:0] buf_l,
    input [3:0] buf_c,

    input [img_width*img_height*bitwidth-1:0]  img,         //图像
    input [weight_width*weight_height*bitwidth-1:0] weight,//权重

    output [bitwidth-1:0] img_cal,
    output [bitwidth-1:0] wei_cal
);


reg [bitwidth-1:0] img_buffer [0:weight_width-1][0:weight_height];
reg [3:0] i,j;
always@(posedge clk_en)begin
    if(!rst_n)begin
        for(i=0;i<weight_height;i=i+1)begin
            for(j=0;j<weight_width;j=j+1)begin
                img_buffer[i][j]=0;
            end
        end
    end
    else begin
        if(conv_on)begin
            for(i=0;i<weight_height;i=i+1)begin
                for(j=0;j<weight_width;j=j+1)begin
                    if (padding_enable) begin
                        if((anchor_l+i)<padding|(anchor_l+i)>img_height|(anchor_c+j)<padding|(anchor_c+j)>img_width)begin
                            img_buffer[i][j]=0;
                        end
                        else begin
                        img_buffer[i][j]=img[((anchor_l+i-padding)*img_width+(anchor_c-padding+j))*bitwidth +:bitwidth];
                        end
                    end
                    else begin
                        img_buffer[i][j]=img[((anchor_l+i)*img_width+(anchor_c+j))*bitwidth +:bitwidth];
                    end
                end
            end
        end
        else begin
            for(i=0;i<weight_height;i=i+1)begin
                for(j=0;j<weight_width;j=j+1)begin
                    img_buffer[i][j]=0;
                end
            end          
        end
    end
end

reg [bitwidth-1:0] weight_buffer [0:weight_width-1][0:weight_height];
reg [3:0] m,n;
always@(*)begin
    if(!rst_n)begin
        for(m=0;m<weight_height;m=m+1)begin
            for(n=0;n<weight_width;n=n+1)begin
                weight_buffer[m][n]=0;
            end
        end
    end
    else begin
        if(conv_on)begin
            for(m=0;m<weight_height;m=m+1)begin
                for(n=0;n<weight_width;n=n+1)begin
                    weight_buffer[m][n]=weight[(m*weight_width+n)*bitwidth+:bitwidth];
                end
            end
        end
        else begin
            for(m=0;m<weight_height;m=m+1)begin
                for(n=0;n<weight_width;n=n+1)begin
                    weight_buffer[m][n]=0;
                end
            end
        end
    end
end

assign img_cal=img_buffer[buf_l][buf_c];
assign wei_cal=weight_buffer[buf_l][buf_c];

endmodule