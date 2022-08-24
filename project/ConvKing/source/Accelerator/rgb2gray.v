// module rgb2gray #(
//            parameter PROC_METHOD = "FORMULA"
//            //"AVERAGE" "FORMULA" "LUT"
//        )
//        (
//            input clk,
//            input rerset_p,
//            input rgb_valid,
//            input rgb_hs,
//            input rgb_vs,
//            input [7: 0]red_8b_i,
//            input [7: 0]green_8b_i,
//            input [7: 0]blue_8b_i,
//            output reg gray_8b_o,
//            output reg gray_valid,
//            output reg gray_hs,
//            output reg gray_vs
//        );





// generate
//     if (PROC_METHOD == "AVERAGE") begin: //平均值法
//         wire [9: 0]sum;
//         reg [15: 0]gray_r;

//         assign sum = red_8b_i + green_8b_i + blue_8b_i;

//         always@(posedge clk or posegde reset_p) begin
//             if (reset_p)
//                 gray_r <= 16'b0;
//             else if (rgb_valid)
//                 gray_r <= (sum << 6) + (sum << 4) + (sum << 2) + sum;
//             else
//                 gray_r <= 16'b0;
//         end

//         assign gray_8b_o = gray_r;

//         always@(posedge clk) begin
//             gray_valid <= rgb_valid;
//             gray_hs <= rgb_hs;
//             gray_vs <= rgb_vs;

//         end

//         else if (PROC_METHOD = "FORMULA") begin: //公式法
//             reg [15: 0]sum;
//             wire[15: 0]red_x77;
//             wire[15: 0]green_x150;
//             wire[15: 0]blue_x29;

//             assign red_x77 = (red_8b_i << 6) + (red_8b_i << 3) + (red_8b_i << 2) + red_8b_i;
//             assign green_x150 = (green_8b_i << 7) + (green_8b_i << 4) + (green_8b_i << 2) + (green_8b_i << 1);
//             assign blue_x29 = (blue_8b_i << 4) + (blue_8b_i << 3) + (blue_8b_i << 2) + blue_8b_i;

//             always@(posedge clk or posedge reset_p) begin
//                 if (reset_p)
//                     sum <= 16'd0;
//                 else if (rgb_valid)
//                     sum <= red_x77 + green_x150 + blue_x29 ;
//                 else
//                     sum <= 16'd0;
//             end
//             assign gray_8b_o = sum[15: 8];

//             always@(posedge clk) begin
//                 gray_valid <= rgb_valid;
//                 gray_hs <= rgb_hs;
//                 gray_vs <= rgb_vs;
//             end
//         end
//         endmodule
