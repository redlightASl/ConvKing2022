`timescale 1ns/1ps
`include "fruit_defines.v"

module uart_decode #(
           parameter CLK_FRE = 50,      //Mhz
           parameter BAUD_RATE = 115200
       )(
           input sys_clk,
           input rst_n,

           input is_count_mode_i,
           input [7: 0] fruit_number,    //number of fruit result
           input is_color_mode_i,
           input [2: 0] color,   //color of fruit result
           input is_detect_mode_i,
           input [2: 0] subject,      //subject of fruit result

           //uart pins
           input uart_rx,
           output uart_tx
       );
wire rx_data_valid;
wire tx_data_ready;
reg tx_data_valid;
reg [7: 0] tx_data;
wire [7: 0] rx_data;

//Control fruit info data output
reg [3: 0] tx_cnt;
reg [7: 0] fruit_data;
always @( * ) begin
    if (is_count_mode_i) begin //show number
        case (fruit_number)
            8'd0: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "0";
                    4'd1:
                        fruit_data = "\r";
                    4'd2:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            8'd1: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "1";
                    4'd1:
                        fruit_data = "\r";
                    4'd2:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            8'd2: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "2";
                    4'd1:
                        fruit_data = "\r";
                    4'd2:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            8'd3: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "3";
                    4'd1:
                        fruit_data = "\r";
                    4'd2:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            8'd4: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "4";
                    4'd1:
                        fruit_data = "\r";
                    4'd2:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            8'd5: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "5";
                    4'd1:
                        fruit_data = "\r";
                    4'd2:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            8'd6: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "6";
                    4'd1:
                        fruit_data = "\r";
                    4'd2:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            8'd7: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "7";
                    4'd1:
                        fruit_data = "\r";
                    4'd2:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            8'd8: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "8";
                    4'd1:
                        fruit_data = "\r";
                    4'd2:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            8'd9: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "9";
                    4'd1:
                        fruit_data = "\r";
                    4'd2:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            8'd10: begin
                case (tx_cnt)
                    4'd0:
                        fruit_data = "1";
                    4'd1:
                        fruit_data = "0";
                    4'd2:
                        fruit_data = "\r";
                    4'd3:
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            default: begin
                fruit_data = " ";
            end
        endcase
    end
    else if (is_color_mode_i) begin //show color
        case (color)
            `COR_RED: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "R";
                    4'd1 :
                        fruit_data = "E";
                    4'd2 :
                        fruit_data = "D";
                    4'd3 :
                        fruit_data = "\r";
                    4'd4 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `COR_ORANGE: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "O";
                    4'd1 :
                        fruit_data = "R";
                    4'd2 :
                        fruit_data = "A";
                    4'd3 :
                        fruit_data = "N";
                    4'd4 :
                        fruit_data = "G";
                    4'd5 :
                        fruit_data = "E";
                    4'd6 :
                        fruit_data = "\r";
                    4'd7 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `COR_YELLOW: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "Y";
                    4'd1 :
                        fruit_data = "E";
                    4'd2 :
                        fruit_data = "L";
                    4'd3 :
                        fruit_data = "L";
                    4'd4 :
                        fruit_data = "O";
                    4'd5 :
                        fruit_data = "W";
                    4'd6 :
                        fruit_data = "\r";
                    4'd7 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `COR_GREEN: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "G";
                    4'd1 :
                        fruit_data = "R";
                    4'd2 :
                        fruit_data = "E";
                    4'd3 :
                        fruit_data = "E";
                    4'd4 :
                        fruit_data = "N";
                    4'd5 :
                        fruit_data = "\r";
                    4'd6 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `COR_PURPLE: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "P";
                    4'd1 :
                        fruit_data = "U";
                    4'd2 :
                        fruit_data = "R";
                    4'd3 :
                        fruit_data = "P";
                    4'd4 :
                        fruit_data = "L";
                    4'd5 :
                        fruit_data = "E";
                    4'd6 :
                        fruit_data = "\r";
                    4'd7 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            default: begin
                fruit_data = " ";
            end
        endcase
    end
    else if (is_detect_mode_i) begin //show kinds
        case (subject)
            `SUB_APPLE: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "A";
                    4'd1 :
                        fruit_data = "P";
                    4'd2 :
                        fruit_data = "P";
                    4'd3 :
                        fruit_data = "L";
                    4'd4 :
                        fruit_data = "E";
                    4'd5 :
                        fruit_data = "\r";
                    4'd6 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `SUB_BANANA: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "B";
                    4'd1 :
                        fruit_data = "A";
                    4'd2 :
                        fruit_data = "N";
                    4'd3 :
                        fruit_data = "A";
                    4'd4 :
                        fruit_data = "N";
                    4'd5 :
                        fruit_data = "A";
                    4'd6 :
                        fruit_data = "\r";
                    4'd7 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `SUB_GRAPE: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "G";
                    4'd1 :
                        fruit_data = "R";
                    4'd2 :
                        fruit_data = "A";
                    4'd3 :
                        fruit_data = "P";
                    4'd4 :
                        fruit_data = "E";
                    4'd5 :
                        fruit_data = "\r";
                    4'd6 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `SUB_PITAYA: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "P";
                    4'd1 :
                        fruit_data = "I";
                    4'd2 :
                        fruit_data = "T";
                    4'd3 :
                        fruit_data = "A";
                    4'd4 :
                        fruit_data = "Y";
                    4'd5 :
                        fruit_data = "A";
                    4'd6 :
                        fruit_data = "\r";
                    4'd7 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `SUB_PEAR: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "P";
                    4'd1 :
                        fruit_data = "E";
                    4'd2 :
                        fruit_data = "A";
                    4'd3 :
                        fruit_data = "R";
                    4'd4 :
                        fruit_data = "\r";
                    4'd5 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `SUB_MANGO: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "M";
                    4'd1 :
                        fruit_data = "A";
                    4'd2 :
                        fruit_data = "N";
                    4'd3 :
                        fruit_data = "G";
                    4'd4 :
                        fruit_data = "O";
                    4'd5 :
                        fruit_data = "\r";
                    4'd6 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `SUB_KIWI: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "K";
                    4'd1 :
                        fruit_data = "I";
                    4'd2 :
                        fruit_data = "W";
                    4'd3 :
                        fruit_data = "I";
                    4'd4 :
                        fruit_data = "\r";
                    4'd5 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            `SUB_ORANGE: begin
                case (tx_cnt)
                    4'd0 :
                        fruit_data = "O";
                    4'd1 :
                        fruit_data = "R";
                    4'd2 :
                        fruit_data = "A";
                    4'd3 :
                        fruit_data = "N";
                    4'd4 :
                        fruit_data = "G";
                    4'd5 :
                        fruit_data = "\r";
                    4'd6 :
                        fruit_data = "\n";
                    default:
                        fruit_data = " ";
                endcase
            end
            default: begin
                fruit_data = " ";
            end
        endcase
    end
    else begin
        fruit_data = " ";
    end
end


localparam UART_IDLE = 3'b001;
localparam UART_SEND = 3'b010;
localparam UART_WAIT = 3'b100;


reg [2: 0] uart_current_state;
reg [2: 0] uart_next_state;
always @(posedge sys_clk or negedge rst_n) begin
    if (!rst_n)
        uart_current_state <= UART_IDLE;
    else
        uart_current_state <= uart_next_state;
end

always @( * ) begin
    case (uart_current_state)
        UART_IDLE: begin
            uart_next_state = UART_SEND;
        end
        UART_SEND: begin
            if (tx_data_valid && tx_data_ready && tx_cnt < 4'd8) begin
                uart_next_state = UART_WAIT;
            end
            else begin
                uart_next_state = UART_SEND;
            end
        end
        UART_WAIT: begin
            uart_next_state = UART_SEND;
        end
        default:
            uart_next_state = UART_IDLE;
    endcase
end


always@(posedge sys_clk or negedge rst_n) begin
    if (!rst_n) begin
        tx_data <= 8'd0;
        tx_cnt <= 4'd0;
        tx_data_valid <= 1'b0;
    end
    else begin
        case (uart_current_state)
            UART_IDLE: begin
                tx_data <= 8'd0;
                tx_cnt <= 4'd0;
                tx_data_valid <= 1'b0;
            end
            UART_SEND: begin
                tx_data <= fruit_data;
                if (tx_data_valid == 1'b1 && tx_data_ready == 1'b1 && tx_cnt < 4'd8) begin
                    tx_cnt <= tx_cnt + 4'd1;
                end
                else if (tx_data_valid && tx_data_ready) begin
                    tx_cnt <= 4'd0;
                    tx_data_valid <= 1'b0;
                end
                else if (~tx_data_valid) begin
                    tx_data_valid <= 1'b1;
                end
            end
            UART_WAIT: begin
                if (rx_data_valid) begin
                    tx_data <= rx_data;
                    tx_data_valid <= 1'b1;
                end
                else if (tx_data_valid && tx_data_ready) begin
                    tx_data_valid <= 1'b0;
                end
            end
            default: begin
                tx_data <= 8'd0;
                tx_cnt <= 4'd0;
                tx_data_valid <= 1'b0;
            end
        endcase
    end
end


wire rx_data_ready;
assign rx_data_ready = 1'b1; //always can receive data
uart_rx #(
            .CLK_FRE(CLK_FRE),
            .BAUD_RATE(BAUD_RATE)
        ) uart_rx_inst (
            .clk (sys_clk ),
            .rst_n (rst_n ),
            .rx_data (rx_data ),
            .rx_data_valid (rx_data_valid ),
            .rx_data_ready (rx_data_ready ),
            .rx_pin (uart_rx )
        );

uart_tx #(
            .CLK_FRE(CLK_FRE),
            .BAUD_RATE(BAUD_RATE)
        ) uart_tx_inst (
            .clk (sys_clk ),
            .rst_n (rst_n ),
            .tx_data (tx_data ),
            .tx_data_valid (tx_data_valid ),
            .tx_data_ready (tx_data_ready ),
            .tx_pin (uart_tx )
        );

endmodule
