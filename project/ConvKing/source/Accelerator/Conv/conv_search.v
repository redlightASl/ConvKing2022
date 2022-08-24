module conv_search #(
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

    output [31:0] anchor_l,
    output [31:0] anchor_c,

    output [3:0]  buf_l,
    output [3:0]  buf_c,
    
    output [3:0] rlt_l,
    output [3:0] rlt_c,

    output chge_rlt_o,
    output chge_rlt_q_o,
    output srh_fin
);

//Delay a clock
reg conv_on_q;
always@(posedge clk_en)begin
    if(!rst_n)begin
        conv_on_q<=0;
    end
    else begin
        conv_on_q<=conv_on;
    end
end

//at the last of the searching, we need some signal to keep the addr
reg rkeep_buf;
reg rkeep_srh;
wire keep_buf;
wire keep_srh;

//the signal of changing addr
wire chge_buf;
wire chge_srh;
wire chge_rlt;
reg rchge_srh;
reg rchge_rlt;
assign chge_srh=rchge_srh;
assign chge_rlt=rchge_rlt;
always@(posedge clk_en)begin
    if(!rst_n)begin
        rchge_srh<=0;
        rchge_rlt<=0;
    end
    else begin
        if(conv_on_q)begin
            rchge_srh<=chge_buf;
            rchge_rlt<=rchge_srh;
        end
        else begin
            rchge_srh<=0;
            rchge_rlt<=0;
        end
    end
end
//Delay a clock
wire chge_buf_q;
reg rchge_buf_q;
assign chge_buf_q=rchge_buf_q;
always@(posedge clk_en)begin
    if(!rst_n)begin
        rchge_buf_q<=0;
    end
    else begin
        if(conv_on_q)begin
            rchge_buf_q<=chge_buf;
        end
        else begin
            rchge_buf_q<=0;
        end
    end
end

wire chge_srh_q;
reg rchge_srh_q;
assign chge_srh_q=rchge_srh_q;
always@(posedge clk_en)begin
    if(!rst_n)begin
        rchge_srh_q<=0;
    end
    else begin
        if(conv_on_q)begin
            rchge_srh_q<=chge_srh;
        end
        else begin
            rchge_srh_q<=0;
        end
    end
end

wire chge_rlt_q;
reg rchge_rlt_q;
assign chge_rlt_q=rchge_rlt_q;
always@(posedge clk_en)begin
    if(!rst_n)begin
        rchge_rlt_q<=0;
    end
    else begin
        if(conv_on_q)begin
            rchge_rlt_q<=chge_rlt;
        end
        else begin
            rchge_rlt_q<=0;
        end
    end
end
assign chge_rlt_o=chge_rlt;
assign chge_rlt_q_o=chge_rlt_q;


//search the addr of the buff
reg [3:0] rbuf_l;
reg [3:0] rbuf_c;
wire chge_buf_l;
assign buf_l=rbuf_l;
assign buf_c=rbuf_c;
assign chge_buf_l=(rbuf_c==weight_width-1)?1:0;
assign chge_buf=(rbuf_l==weight_height-1)&(rbuf_c==weight_width-2)?1:0;
always @(posedge clk_en) begin
    if(!rst_n)begin
        rbuf_l<=0;
        rbuf_c<=0;
    end
    else begin
        if(conv_on_q)begin
            if(keep_srh)begin
                rbuf_l<=rbuf_l;
                rbuf_c<=rbuf_c;
            end
            else begin
                if(~chge_srh&chge_srh_q)begin
                    rbuf_l<=0;
                    rbuf_c<=0;
                end
                else if(chge_srh&~chge_srh_q)begin
                    rbuf_l<=rbuf_l;
                    rbuf_c<=rbuf_c;
                end
                else begin
                    if(chge_buf_l)begin
                        rbuf_l<=rbuf_l+1;
                        rbuf_c<=0;
                    end
                    else begin
                        rbuf_l<=rbuf_l;
                        rbuf_c<=rbuf_c+1;
                    end
                end
            end
        end
        else begin
            rbuf_l<=0;
            rbuf_c<=0;
        end
    end
end

//search the addr of the result arrays
reg [3:0] rrlt_l;
reg [3:0] rrlt_c;
wire chge_rlt_l;
wire keep;
assign rlt_l=rrlt_l;
assign rlt_c=rrlt_c;
assign chge_rlt_l=(rrlt_c==result_width-1)?1:0;
assign keep=(rrlt_c==result_width-1)&(rrlt_l==result_height-1)&chge_buf?1:0;
assign srh_fin=(rrlt_l==result_height)?1:0;
always @(posedge clk_en) begin
    if(!rst_n)begin
       rrlt_l<=0;
       rrlt_c<=0; 
    end
    else begin
        if(conv_on_q)begin
            if(~chge_rlt&chge_rlt_q&chge_rlt_l)begin
                    rrlt_l<=rrlt_l+1;
                    rrlt_c<=0;
            end
            else if(chge_rlt&~chge_rlt_q&chge_rlt_l)begin
                    rrlt_l<=rrlt_l;
                    rrlt_c<=rrlt_c;
            end
            else if(~chge_rlt&chge_rlt_q&!chge_rlt_l)begin
                    rrlt_l<=rrlt_l;
                    rrlt_c<=rrlt_c+1;
            end
            else begin
                    rrlt_l<=rrlt_l;
                    rrlt_c<=rrlt_c;
            end
        end
        else begin
            rrlt_l<=0;
            rrlt_c<=0;
        end
    end
end

//search the addr of the top left point
reg [31:0] ranchor_l;
reg [31:0] ranchor_c;
assign anchor_l=ranchor_l;
assign anchor_c=ranchor_c;
always@(posedge clk_en)begin
    if(!rst_n)begin
        ranchor_l<=0;
        ranchor_c<=0;
    end
    else begin
        if(conv_on_q)begin
            if(keep_buf)begin
                ranchor_l<=ranchor_l;
                ranchor_c<=ranchor_c;
            end
            else begin
                if(~chge_buf&chge_buf_q&chge_rlt_l)begin
                    ranchor_l<=ranchor_l+stride;
                    ranchor_c<=0;
                end
                else if(~chge_buf&chge_buf_q&!chge_rlt_l)begin
                    ranchor_l<=ranchor_l;
                    ranchor_c<=ranchor_c+stride;
                end
                else if(chge_buf&~chge_buf_q&!chge_rlt_l)begin
                    ranchor_l<=ranchor_l;
                    ranchor_c<=ranchor_c;
                end
                else begin
                    ranchor_l<=ranchor_l;
                    ranchor_c<=ranchor_c;
                end
            end
        end
        else begin
            ranchor_l<=0;
            ranchor_c<=0;
        end
    end
end

//effective immediately
assign keep_buf=rkeep_buf;
always@(*)begin
    if(!rst_n)begin
        rkeep_buf=0;
    end
    else begin
        if(conv_on_q)begin
            if(keep)begin
                rkeep_buf=1;
            end
            else begin
                rkeep_buf=rkeep_buf;
            end
        end
        else begin
            rkeep_buf=0;
        end
    end
end
//effective after a clock
assign keep_srh=rkeep_srh;
always@(posedge clk_en)begin
    if(!rst_n)begin
        rkeep_srh<=0;
    end
    else begin
        if(conv_on_q)begin
            if(keep)begin
                rkeep_srh<=1;
            end
            else begin
                rkeep_srh<=rkeep_srh;
            end
        end
        else begin
            rkeep_srh<=0;
        end
    end
end
endmodule