// `timescale 1ns / 1ps

// module linear#(
//            parameter DATABITWIDTH = 8,
//            parameter DATALENGTH = 28 * 28,
//            parameter OUTFEATURES = 20,
//            parameter BIASBITWIDTH = 8
//        )(
//            input clk,
//            input rstn,
//            input flag,  //串口中断标志位
//            input signed [DATABITWIDTH - 1 : 0] in_features,
//            input signed [BIASBITWIDTH * OUTFEATURES - 1 : 0] bias,
//            output reg signed [OUTFEATURES * DATABITWIDTH * 2 - 1 : 0] outfeatures,
//            output reg en_out
//        );

// wire signed [BIASBITWIDTH - 1 : 0] biasArray[0 : OUTFEATURES - 1];
// wire signed [2 * DATABITWIDTH - 1 : 0] intermediateArray[0 : OUTFEATURES - 1];
// reg signed [DATABITWIDTH - 1 : 0] infeatures;
// reg [DATABITWIDTH - 1 : 0] weight_reg[0 : OUTFEATURES - 1][0 : DATALENGTH - 1];
// reg signed [2 * DATABITWIDTH - 1 : 0] resultArray[0 : OUTFEATURES - 1];
// integer cnt;
// wire clk_2;

// /*
//     //分频乘法器和加法器时钟
//     clk_wiz_0 clk_div
//   (
//   // Clock out ports  
//   .clk_out1(clk_2),
//   // Status and control signals               
//   .reset(rstn), 
//  // Clock in ports
//   .clk_in1(clk)
//   );
// */

// genvar i;
// generate //存偏置
//     for (i = 0; i < OUTFEATURES; i = i + 1) begin
//         assign biasArray[i] = bias[(i + 1) * BIASBITWIDTH - 1 : i * BIASBITWIDTH];
//     end
// endgenerate

// //计数器
// always @(posedge clk or negedge rstn) begin
//     if (!rstn)
//         cnt <= 16'b0;
//     else if (cnt == (DATALENGTH - 1)) begin
//         cnt <= 16'b0;
//         en_out <= 1'b1; //此时输出有效
//     end
//     else begin
//         cnt <= cnt + 1'b1;
//         en_out <= 1'b0; //每DATALENGTH个时钟周期输出有效
//     end
// end

// //接受串口数据传输
// always @(posedge clk) begin
//     if (flag == 1'b1)
//         infeatures <= in_features;
// end

// /*
//     //计算本帧传输第cnt个数据与权重相作用之积
//     genvar j;
//     generate
//         for(j = 0; j < OUTFEATURES; j = j + 1)begin
 
//             mult_gen_0 mult
//             (
//             .CLK(clk),
//             .A(infeatures),
//             .B(weight_reg[j][cnt]),
//             .P(intermediateArray[j])
//             );
//             //assign intermediateArray[j] = infeatures * weight_reg[j][cnt];//cnt的初始问题
//         end
//     endgenerate
// */

// //每个时钟上升沿加一组数据
// integer n;
// always @(posedge clk) begin
//     for (n = 0; n < OUTFEATURES; n = n + 1) begin
//         if (n == (OUTFEATURES - 1))
//             resultArray[n] <= (resultArray[n] + intermediateArray[n] + biasArray[n]);
//         else
//             resultArray[n] <= (resultArray[n] + intermediateArray[n]);
//     end
// end

// //        c_addsub_0
// //        (
// //        .A(resultArray[n]),
// //        .B(intermediateArray[n]),
// //        .CLK(clk_2),
// //        .S(resultArray[n])
// //        );

// //    begin
// //    if(cnt == OUTFEATURES - 1) begin
// //        c_addsub_0
// //        (
// //        .A(resultArray[n]),
// //        .B(biasArray[n]),
// //        .CLK(clk_2),
// //        .S(resultArray[n])
// //        );
// //    end
// //    end

// //初始化 resultArray
// integer k;
// always @(posedge clk or negedge rstn) begin
//     if (!rstn) begin
//         for (k = 0; k < OUTFEATURES; k = k + 1) begin
//             resultArray[k] <= {2 * DATABITWIDTH{1'b0}};
//         end
//     end
// end

// //初始化intermediateArray
// //    integer m;
// //    always @(posedge clk or negedge rstn) begin
// //        if(!rstn) begin
// //            for(m =0; m < OUTFEATURES; m = m + 1) begin
// //                intermediateArray[m] <= {2 * DATABITWIDTH{1'b0}};
// //            end
// //        end
// //    end

// //赋值权重
// integer s, x;
// always @(posedge clk or negedge rstn) begin
//     if (!rstn) begin
//         for (s = 0; s < OUTFEATURES; s = s + 1) begin
//             for (x = 0; x < DATALENGTH; x = x + 1) begin
//                 weight_reg[s][x] <= {DATABITWIDTH{1'b1}};
//             end
//         end
//     end
// end

// //复位
// always @(posedge clk or negedge rstn) begin
//     if (!rstn) begin
//         en_out <= 1'b0;
//         outfeatures <= {OUTFEATURES * DATABITWIDTH * 2{1'b0}};
//     end
// end
// //输出处理得到特征值
// integer l;
// always @(posedge clk) begin
//     if (en_out == 1) begin //得等DATALENGTH个时钟周期后得到最终数值
//         for (l = 0; l < OUTFEATURES; l = l + 1) begin
//             outfeatures[(l * DATABITWIDTH) +: DATABITWIDTH] <= resultArray[l];
//         end
//     end
// end

// endmodule
