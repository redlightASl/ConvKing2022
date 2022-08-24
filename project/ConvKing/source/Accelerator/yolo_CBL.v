`timescale 1ns/1ps

module yolo_cbl(

       );



conv_top #(
             .weight_width ( 2 ),
             .weight_height ( 2 ),
             .img_width ( 4 ),
             .img_height ( 4 ),
             .padding_enable ( 0 ),
             .padding ( 0 ),
             .stride ( 2 ),
             .bitwidth ( 3 ),
             //  .result_width (  ),
             //  .result_height (  ),
             .expand ( 1 ))
         u_conv_top(
             //ports
             .clk_en ( ),
             .rst_n ( ),
             .conv_en ( ),
             .img ( ),
             .weight ( ),
             .bias ( ),
             .result ( ),
             .conv_fin ( )
         );

relu #(
         .BITWIDTH ( 8 ),
         //  .THRESSHOLD (  ),
         .MAX_VAL ( 6 ))
     u_relu(
         //ports
         .in_data ( ),
         .result ( )
     );

endmodule
