`timescale 1ns/1ps

module relu #(
           parameter BITWIDTH = 8,
           parameter THRESSHOLD = {BITWIDTH{1'b0}},
           parameter MAX_VAL = 6
       )(
           input wire [BITWIDTH - 1 : 0] in_data, //signed data
           output reg [BITWIDTH - 1 : 0] result //signed data
       );

always @( * ) begin
    if (in_data > THRESSHOLD) begin
        if (in_data < MAX_VAL) begin
            result = in_data;
        end
        else begin
            result = MAX_VAL;
        end
    end
    else begin
        result = THRESSHOLD;
    end
end
endmodule
