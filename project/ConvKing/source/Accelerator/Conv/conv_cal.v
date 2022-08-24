module conv_cal #(
           parameter weight_width = 2,
           parameter weight_height = 2,

           parameter img_width = 4,
           parameter img_height = 4,

           parameter padding_enable = 0,
           parameter padding = 0,

           parameter stride = 1,
           parameter bitwidth = 3,
           parameter result_width = (img_width - weight_width + 2 * padding) / stride + 1,
           parameter result_height = (img_height - weight_height + 2 * padding) / stride + 1,
           parameter expand = 1     //expand the bitwidth of result
       ) (
           input clk_en,
           input rst_n,
           input conv_on,
           input srh_fin,

           input chge_rlt,
           input chge_rlt_q,

           input [3: 0] rlt_l,
           input [3: 0] rlt_c,

           input [bitwidth - 1: 0] bias,
           input [bitwidth - 1: 0] img_cal,
           input [bitwidth - 1: 0] wei_cal,

           output [expand * 2 * result_width * result_height * bitwidth - 1: 0] result
       );


//save the result from the mult-acc ip
wire [23: 0] data_link;
wire [15: 0] addr;
assign addr = rlt_l * result_width + rlt_c;  //the one-dimensional address of the result
reg [23: 0] rdata [0: result_width * result_height - 1];
reg [3: 0] i;
always @(posedge clk_en) begin
    if (!rst_n) begin
        for (i = 0;i < result_width * result_height;i = i + 1) begin
            rdata[i] = 0;
        end
    end
    else if (conv_on) begin
        if (chge_rlt_q) begin
            for (i = 0;i < result_width * result_height;i = i + 1) begin
                rdata[i] = rdata[i];
            end
        end
        else begin
            rdata[addr] = data_link;
        end
    end
    else begin
        for (i = 0;i < result_width * result_height;i = i + 1) begin
            rdata[i] = rdata[i];
        end
    end
end

//switch parallel to serial
generate
    genvar j;
    for (j = 0;j < result_width * result_height;j = j + 1) begin
        assign result[j * (expand * 2 * bitwidth) +: (expand * 2 * bitwidth)] = rdata[j][(expand * 2 * bitwidth) - 1: 0] + {{(expand * 2 - 1) * bitwidth{1'b0}}, bias};
    end
endgenerate

//the signal to reload the data_o to zero
wire reload;
assign reload = chge_rlt | srh_fin;


mult_add u_mult_add(
             //ports
             .sys_clk_i ( clk_en ),
             .sys_rst_n_i ( ~rst_n ),
             .sys_is_add_i ( 1'b0 ),
             .am_num_A0_i ( img_cal ),
             .am_num_A1_i ( wei_cal ),
             .am_num_B0_i ( conv_on ),
             .am_num_B1_i ( reload ),
             .am_mult_o ( data_link )
         );


endmodule
