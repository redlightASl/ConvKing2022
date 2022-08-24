module histogram #(
           parameter IMG_DATA_BITWIDTH = 16,
           parameter IMG_WIDTH = 1024,
           parameter IMG_HEIGHT = 768,
           parameter HSV_H_LEVEL = 9,
           parameter HSV_S_LEVEL = 9,
           parameter HSV_V_LEVEL = 8
       )(
           //system
           input sys_clk_i,
           input sys_rst_n_i,

           //input image data
           input [HSV_H_LEVEL - 1: 0] hsv_h,
           input [HSV_S_LEVEL - 1: 0] hsv_s,
           input [HSV_V_LEVEL - 1: 0] hsv_v,

           //control sync timing trans
           input hsync_i,
           input vsync_i,
           input de_i,

           output wire [31: 0] histogram_data_h_o,
           output wire [31: 0] histogram_data_s_o,
           output wire [31: 0] histogram_data_v_o
       );
localparam IMAGE_PIXEL_NUM = IMG_WIDTH * IMG_HEIGHT;

//edge detect for vsync
wire vsync_negedge;
wire vsync_posedge;
reg vsync_d0;

assign vsync_negedge = vsync_i & (~vsync_i);
assign vsync_posedge = (~vsync_i) & vsync_i;
always @(posedge sys_clk_i or negedge sys_rst_n_i) begin
    if (!sys_rst_n_i) begin
        vsync_d0 <= 1'b0;
    end
    else begin
        vsync_d0 <= vsync_i;
    end
end

reg ram_output_en;
wire [31: 0] histogram_data_h_reg; //red histogram output reg
wire [31: 0] histogram_data_s_reg; //green histogram output reg
wire [31: 0] histogram_data_v_reg; //blue histogram output reg

//hsync posedge: frame done
//allow result to output
assign histogram_data_h_o = (ram_output_en) ? histogram_data_h_reg : 32'd0;
assign histogram_data_s_o = (ram_output_en) ? histogram_data_s_reg : 32'd0;
assign histogram_data_v_o = (ram_output_en) ? histogram_data_v_reg : 32'd0;

always @(posedge sys_clk_i or negedge sys_rst_n_i) begin
    if (!sys_rst_n_i) begin
        ram_output_en <= 1'b0;
    end
    else begin
        if (vsync_posedge) begin
            ram_output_en <= 1'b1;
        end
        else begin
            ram_output_en <= 1'b0;
        end
    end
end

//hsync negedge: frame start
//init RAM = 0
reg ram_clr_en_h;
reg ram_clr_en_s;
reg ram_clr_en_v;
always @(posedge sys_clk_i or negedge sys_rst_n_i) begin
    if (!sys_rst_n_i) begin
        ram_clr_en_h <= 1'b0;
        ram_clr_en_s <= 1'b0;
        ram_clr_en_v <= 1'b0;
    end
    else begin
        if (vsync_negedge) begin
            ram_clr_en_h <= 1'b1;
            ram_clr_en_s <= 1'b1;
            ram_clr_en_v <= 1'b1;
        end
        else begin
            ram_clr_en_h <= 1'b0;
            ram_clr_en_s <= 1'b0;
            ram_clr_en_v <= 1'b0;
        end
    end
end

//foreach RAM address when read or write 'Port B'
reg [HSV_H_LEVEL - 1: 0] ram_read_addr_h;
reg [HSV_S_LEVEL - 1: 0] ram_read_addr_s;
reg [HSV_V_LEVEL - 1: 0] ram_read_addr_v;

always @(posedge sys_clk_i or negedge sys_rst_n_i) begin
    if (!sys_rst_n_i) begin
        ram_read_addr_h <= {HSV_H_LEVEL{1'b0}};
        ram_read_addr_s <= {HSV_S_LEVEL{1'b0}};
        ram_read_addr_v <= {HSV_V_LEVEL{1'b0}};
    end
    else begin
        if (ram_output_en) begin
            ram_read_addr_h <= ram_read_addr_h + 1'b1;
            ram_read_addr_s <= ram_read_addr_s + 1'b1;
            ram_read_addr_v <= ram_read_addr_v + 1'b1;
        end
        else if (ram_clr_en_h | ram_clr_en_s | ram_clr_en_v) begin
            ram_read_addr_h <= ram_read_addr_h + 1'b1;
            ram_read_addr_s <= ram_read_addr_s + 1'b1;
            ram_read_addr_v <= ram_read_addr_v + 1'b1;
        end
        else begin
            ram_read_addr_h <= {HSV_H_LEVEL{1'b0}};
            ram_read_addr_s <= {HSV_S_LEVEL{1'b0}};
            ram_read_addr_v <= {HSV_V_LEVEL{1'b0}};
        end
    end
end


// histogram_ram_r histogram_ram_r_inst (
//                     .a_clk(sys_clk_i),
//                     .b_clk(sys_clk_i),
//                     .a_rst(!sys_rst_n_i),
//                     .b_rst(1'b0),

//                     .a_addr(rgb_i[15: 11]), // input [4:0]
//                     .a_wr_en(de_i),
//                     .a_wr_data(ram_read_data_r + 1), // input [31:0]
//                     .a_rd_data(ram_read_data_r), // output [31:0]

//                     .b_addr(ram_read_addr_r), // input [4:0]
//                     .b_wr_en(ram_clr_en_r),
//                     .b_wr_data(32'd0), // input [31:0]
//                     .b_rd_data(histogram_data_red_reg) // output [31:0]
//                 );
// histogram_ram_g histogram_ram_g_inst (
//                     .a_clk(sys_clk_i),
//                     .b_clk(sys_clk_i),
//                     .a_rst(!sys_rst_n_i),
//                     .b_rst(1'b0),

//                     .a_addr(rgb_i[10: 5]),             // input [5:0]
//                     .a_wr_en(de_i),           // input
//                     .a_wr_data(ram_read_data_g + 1),       // input [31:0]
//                     .a_rd_data(ram_read_data_g),       // output [31:0]

//                     .b_addr(ram_read_addr_g),             // input [5:0]
//                     .b_wr_en(ram_clr_en_r),           // input
//                     .b_wr_data(32'd0),       // input [31:0]
//                     .b_rd_data(histogram_data_green_reg)     // output [31:0]
//                 );
// histogram_ram_b histogram_ram_b_inst (
//                     .a_clk(sys_clk_i),
//                     .b_clk(sys_clk_i),
//                     .a_rst(!sys_rst_n_i),
//                     .b_rst(1'b0),

//                     .a_addr(rgb_i[4: 0]),             // input [4:0]
//                     .a_wr_en(de_i),           // input
//                     .a_wr_data(ram_read_data_b + 1),       // input [31:0]
//                     .a_rd_data(ram_read_data_b),       // output [31:0]

//                     .b_addr(ram_read_addr_b),             // input [4:0]
//                     .b_wr_en(ram_clr_en_r),           // input
//                     .b_wr_data(32'd0),       // input [31:0]
//                     .b_rd_data(histogram_data_blue_reg)     // output [31:0]
//                 );


wire [31: 0] ram_read_data_h;
wire [31: 0] ram_read_data_s;
wire [31: 0] ram_read_data_v;

histogram_ram_h histogram_ram_h_inst (
                    .a_clk(sys_clk_i),
                    .b_clk(sys_clk_i),
                    .a_rst(!sys_rst_n_i),
                    .b_rst(1'b0),

                    .a_addr(hsv_h),           // input [8:0]
                    .a_wr_data(ram_read_data_h + 1),     // input [31:0]
                    .a_rd_data(ram_read_data_h),     // output [31:0]
                    .a_wr_en(de_i),         // input

                    .b_addr(ram_read_addr_h),           // input [8:0]
                    .b_wr_data(32'd0),     // input [31:0]
                    .b_rd_data(histogram_data_h_reg),     // output [31:0]
                    .b_wr_en(ram_clr_en_h)        // input
                );
histogram_ram_s histogram_ram_s_inst (
                    .a_clk(sys_clk_i),
                    .b_clk(sys_clk_i),
                    .a_rst(!sys_rst_n_i),
                    .b_rst(1'b0),

                    .a_addr(hsv_s),           // input [8:0]
                    .a_wr_data(ram_read_data_s + 1),     // input [31:0]
                    .a_rd_data(ram_read_data_s),     // output [31:0]
                    .a_wr_en(de_i),         // input

                    .b_addr(ram_read_addr_s),           // input [8:0]
                    .b_wr_data(32'd0),     // input [31:0]
                    .b_rd_data(histogram_data_s_reg),     // output [31:0]
                    .b_wr_en(ram_clr_en_s)        // input
                );
histogram_ram_v histogram_ram_v_inst (
                    .a_clk(sys_clk_i),
                    .b_clk(sys_clk_i),
                    .a_rst(!sys_rst_n_i),
                    .b_rst(1'b0),

                    .a_addr(hsv_v),           // input [8:0]
                    .a_wr_data(ram_read_data_v + 1),     // input [31:0]
                    .a_rd_data(ram_read_data_v),     // output [31:0]
                    .a_wr_en(de_i),         // input

                    .b_addr(ram_read_addr_v),           // input [8:0]
                    .b_wr_data(32'd0),     // input [31:0]
                    .b_rd_data(histogram_data_v_reg),     // output [31:0]
                    .b_wr_en(ram_clr_en_v)        // input
                );

endmodule
