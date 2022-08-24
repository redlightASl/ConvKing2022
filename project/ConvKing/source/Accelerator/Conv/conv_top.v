module conv_top #(
    parameter weight_width = 2,        
    parameter weight_height = 2,        

    parameter img_width = 4,            
    parameter img_height = 4,           
    
    parameter padding_enable = 0,      
    parameter padding = 0,              
    
    parameter stride = 2,               
    parameter bitwidth = 3,             
    parameter result_width = (img_width-weight_width+2*padding)/stride+1,       
    parameter result_height = (img_height-weight_height+2*padding)/stride+1,     
    parameter expand = 1        //expand the bitwidth of result 
    
)(
    input clk_en,
    input rst_n,   
    input conv_en,              //if 1,the conv is on

    input [img_width*img_height*bitwidth-1:0]  img,  
    input [weight_width*weight_height*bitwidth-1:0] weight, 
    input [bitwidth-1:0] bias,                              

    output [expand*2*result_width*result_height*bitwidth-1:0]  result,  
    output conv_fin //if 1, the result of the conv is correct
);


//the addr of the buff
wire [31:0] anchor_l;   //the addr of the top left point
wire [31:0] anchor_c;
wire [3:0]  buf_l;      //the addr of the buf
wire [3:0]  buf_c;
wire [3:0] rlt_l;       //the addr of the result arrays
wire [3:0] rlt_c;

wire [bitwidth-1:0] img_cal;        //the data for calcualtion
wire [bitwidth-1:0] wei_cal;

wire chge_rlt;                      //change the addr of the result array
wire chge_rlt_q;                    //delay a clk
wire srh_fin;                       //the end of the searching

reg rconv_on;                       //the reg of the conv_on
reg rconv_fin;
wire conv_on;

//FSM block
localparam IDLE = 2'b01;        
localparam BUSY = 2'b10;

reg [1:0] cur_state;
reg [1:0] next_state;

always @(posedge clk_en) begin
    if(!rst_n)begin
        cur_state<=IDLE;
    end
    else begin
        cur_state<=next_state;
    end
end

always@(*)begin
    case(cur_state)
        IDLE:begin
           if(conv_en)begin
              next_state=BUSY; 
           end 
           else begin
              next_state=IDLE;
           end
        end
        BUSY:begin
            if (srh_fin) begin
                next_state=IDLE;
            end
            else begin
                next_state=BUSY;
            end
        end
        default:begin
            if(conv_en)begin
                next_state=BUSY;
            end
            else begin
                next_state=IDLE;
            end
        end
    endcase
end


assign conv_on=rconv_on;
assign conv_fin=rconv_fin;
always @(posedge clk_en) begin
    if(!rst_n)begin
        rconv_on<=0;
        rconv_fin<=0;
    end
    else begin
        case(cur_state)
            IDLE:begin
                rconv_on<=0;
                rconv_fin<=1;
            end
            BUSY:begin
                rconv_on<=1;
                rconv_fin<=0;
            end
            default:begin
                rconv_on<=0;
                rconv_fin<=1;
            end
        endcase
    end
end

//buff block
conv_buffer #(
    weight_width,         
    weight_height,       
    img_width,            
    img_height,           
    padding_enable,       
    padding,              
    stride,               
    bitwidth,             
    result_width,       
    result_height,     
    expand       
)conv_buffer_inst(
    .clk_en     (clk_en),
    .rst_n      (rst_n),
    .conv_on    (conv_on),

    .anchor_l   (anchor_l),
    .anchor_c   (anchor_c),

    .buf_l      (buf_l),
    .buf_c      (buf_c),

    .img        (img),
    .weight     (weight),

    .img_cal    (img_cal),
    .wei_cal    (wei_cal)
);

//searching addr block
conv_search #(
    weight_width,         
    weight_height,       
    img_width,            
    img_height,           
    padding_enable,       
    padding,              
    stride,               
    bitwidth,             
    result_width,       
    result_height,     
    expand       
)conv_search_inst(
    .clk_en     (clk_en),
    .rst_n      (rst_n),
    .conv_on    (conv_on),

    .anchor_l   (anchor_l),
    .anchor_c   (anchor_c),

    .buf_l      (buf_l),
    .buf_c      (buf_c),

    .rlt_l      (rlt_l),
    .rlt_c      (rlt_c),

    .chge_rlt_o   (chge_rlt),
    .chge_rlt_q_o (chge_rlt_q),
    .srh_fin    (srh_fin)
);

//calculating block
conv_cal#(
    weight_width,         
    weight_height,       
    img_width,            
    img_height,           
    padding_enable,       
    padding,              
    stride,               
    bitwidth,             
    result_width,       
    result_height,     
    expand     
)conv_cal_inst(
    .clk_en         (clk_en),
    .rst_n          (rst_n),
    .conv_on        (conv_on),
    .srh_fin        (srh_fin),

    .chge_rlt       (chge_rlt),
    .chge_rlt_q     (chge_rlt_q),

    .rlt_l          (rlt_l),
    .rlt_c          (rlt_c),

    .bias           (bias),
    .img_cal        (img_cal),
    .wei_cal        (wei_cal),
    .result         (result)
);

endmodule