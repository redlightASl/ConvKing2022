module color_det_top #(
           parameter KNN_INPUT_BITWIDTH = 16,
           parameter KNN_INPUT_IMG_WIDTH = 1024,   //frame size 1024 * 768 * 16 / 64
           parameter KNN_INPUT_IMG_HEIGHT = 768,
           parameter KNN_INPUT_IMAGE_NUMBER = 3
       )(
           //system
           input sys_clk,
           input sys_rst_n,

           //input data
           input [KNN_INPUT_BITWIDTH - 1: 0] rgb_i,   //input data
           input hsync_i,   //control sync timing trans
           input vsync_i,
           input de_i,

           //output data
           output [2: 0] knn_color
       );
localparam PIXEL_NUM = KNN_INPUT_IMG_WIDTH * KNN_INPUT_IMG_HEIGHT;
localparam KNN_PIX_CNT_BITWIDTH = 24;

reg img_done;













wire [8: 0]	hsv_h;
wire [8: 0]	hsv_s;
wire [7: 0]	hsv_v;
wire hsv_vs;
wire hsv_hs;
wire hsv_de;

rgb2hsv u_rgb2hsv(
            .clk ( sys_clk ),
            .reset_n ( sys_rst_n ),

            .rgb_r ( {rgb_i[15: 11], 3'd0} ),
            .rgb_g ( {rgb_i[10: 5], 2'd0} ),
            .rgb_b ( {rgb_i[4: 0], 3'd0} ),
            .vs ( vsync_i ),
            .hs ( hsync_i ),
            .de ( de_i ),

            .hsv_h ( hsv_h ),
            .hsv_s ( hsv_s ),
            .hsv_v ( hsv_v ),
            .hsv_vs ( hsv_vs ),
            .hsv_hs ( hsv_hs ),
            .hsv_de ( hsv_de )
        );


wire [31: 0] histogram_data_h_o;
wire [31: 0] histogram_data_s_o;
wire [31: 0] histogram_data_v_o;

histogram #(
              .IMG_DATA_BITWIDTH ( KNN_INPUT_BITWIDTH ),
              .IMG_WIDTH ( KNN_INPUT_IMG_WIDTH ),
              .IMG_HEIGHT ( KNN_INPUT_IMG_HEIGHT ),
              .HSV_H_LEVEL ( 9 ),
              .HSV_S_LEVEL ( 9 ),
              .HSV_V_LEVEL ( 8 )
          )
          u_histogram (
              .sys_clk_i ( sys_clk ),
              .sys_rst_n_i ( sys_rst_n ),
              .hsv_h ( hsv_h ),
              .hsv_s ( hsv_s ),
              .hsv_v ( hsv_v ),
              .hsync_i ( hsv_hs ),
              .vsync_i ( hsv_vs ),
              .de_i ( hsv_de ),

              .histogram_data_h_o ( histogram_data_h_o ),
              .histogram_data_s_o ( histogram_data_s_o ),
              .histogram_data_v_o ( histogram_data_v_o )
          );

















// knn_top();

assign knn_color = 3'b100;

endmodule
