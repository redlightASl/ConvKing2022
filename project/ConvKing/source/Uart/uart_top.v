`timescale 1ns/1ps

module uart_top #(
           parameter CLK_FRE = 50,   //Mhz
           parameter BAUD_RATE = 115200
       )
       (
           input sys_clk,
           input uart_fifo_write_clk,
           input rst_n,

           //input control
           input write_req,   //keep '1' until read_req_ack = '1'
           output reg write_req_ack,   //write request response
           input write_en,   //write request for one data
           input [31: 0] write_data,

           //uart pins
           input uart_rx,
           output uart_tx
       );
//recv FSM
localparam RECV_IDLE = 4'b0001;
localparam RECV_WAIT = 4'b0010;
localparam RECV_READ = 4'b0100;
localparam RECV_WRITE = 4'b1000;

reg [3: 0] current_state;
reg [3: 0] next_state;
always @(posedge uart_fifo_write_clk or negedge rst_n) begin
    if (!rst_n)
        current_state <= RECV_IDLE;
    else
        current_state <= next_state;
end

always @( * ) begin
    case (current_state)
        RECV_IDLE: begin
            next_state = RECV_READ;
        end
        RECV_WAIT: begin
            if (write_req) begin
                next_state = RECV_READ;
            end
            else begin
                next_state = RECV_WAIT;
            end
        end
        RECV_READ: begin
            if (write_en) begin
                next_state = RECV_WRITE;
            end
            else begin
                next_state = RECV_WAIT;
            end
        end
        RECV_WRITE: begin
            if(!write_en) begin
                next_state = RECV_READ;
            end
            else begin
                next_state = RECV_WRITE;
            end
        end
        default: begin
            next_state = RECV_IDLE;
        end
    endcase
end

reg write_fifo_en;
reg [31: 0] write_fifo_data;
always @(posedge uart_fifo_write_clk or negedge rst_n) begin
    if (!rst_n) begin
        write_req_ack <= 1'b0;
        write_fifo_en <= 1'b0;
        write_fifo_data <= 32'b0;
    end
    else begin
        case (current_state)
            RECV_IDLE: begin
                write_req_ack <= 1'b0;
                write_fifo_en <= 1'b0;
                write_fifo_data <= 32'b0;
            end
            RECV_WAIT: begin
                write_req_ack <= 1'b0;
                write_fifo_en <= 1'b0;
                write_fifo_data <= 32'b0;
            end
            RECV_READ: begin
                write_req_ack <= 1'b1;
                write_fifo_en <= 1'b0;
                write_fifo_data <= 32'b0;
            end
            RECV_WRITE: begin
                write_req_ack <= 1'b0;
                write_fifo_en <= 1'b1;
                write_fifo_data <= write_data;
            end
            default: begin
                write_req_ack <= 1'b1;
                write_fifo_en <= 1'b0;
                write_fifo_data <= 32'b0;
            end
        endcase
    end
end

wire read_fifo_clear;
wire write_fifo_clear;
wire read_empty_flag;
reg read_en;
wire [7: 0] read_data;
afifo_32i_8o_64depth uart_write_fifo (
                         .wr_clk(uart_fifo_write_clk),
                         .wr_rst(write_fifo_clear),
                         .wr_en(write_fifo_en),
                         .wr_data(write_fifo_data),
                         .wr_full(),
                         .wr_water_level(),
                         .almost_full(),

                         .rd_clk(sys_clk),
                         .rd_rst(read_fifo_clear),
                         .rd_en(read_en),
                         .rd_data(read_data),
                         .rd_empty(read_empty_flag),
                         .rd_water_level(),
                         .almost_empty()
                     );

//uart FSM
localparam UART_IDLE = 4'b0001;
localparam UART_WAIT = 4'b0010;
localparam UART_READ = 4'b0100;
localparam UART_SEND = 4'b1000;


wire rx_data_valid;

wire tx_data_ready;
reg tx_data_valid;

reg [7: 0] tx_data;
wire [7: 0] rx_data;

reg [3: 0] uart_current_state;
reg [3: 0] uart_next_state;
always @(posedge sys_clk or negedge rst_n) begin
    if (!rst_n)
        uart_current_state <= UART_IDLE;
    else
        uart_current_state <= uart_next_state;
end

always @( * ) begin
    case (uart_current_state)
        UART_IDLE: begin
            uart_next_state = UART_WAIT;
        end
        UART_WAIT: begin
            if (!read_empty_flag) begin
                uart_next_state = UART_READ;
            end
            else begin
                uart_next_state = UART_WAIT;
            end
        end
        UART_READ: begin
            if (tx_data_valid && tx_data_ready) begin
                uart_next_state = UART_SEND;
            end
            else if (read_empty_flag) begin
                uart_next_state = UART_WAIT;
            end
            else begin
                uart_next_state = UART_READ;
            end
        end
        UART_SEND: begin
            uart_next_state = UART_READ;
        end
        default:
            uart_next_state = UART_IDLE;
    endcase
end

always @(posedge sys_clk or negedge rst_n) begin
    if (!rst_n) begin
        read_en <= 1'b0;
        tx_data <= 8'b0;
        tx_data_valid <= 1'b0;
    end
    else begin
        case (uart_current_state)
            UART_WAIT: begin
                read_en <= 1'b0;
                tx_data <= 8'b0;
                tx_data_valid <= 1'b0;
            end 
            UART_READ: begin
                read_en <= 1'b1;
                tx_data <= read_data;
                tx_data_valid <= 1'b0;
            end
            UART_SEND: begin
                read_en <= 1'b0;
                tx_data <= read_data;
                if(!tx_data_valid) begin 
                    tx_data_valid <= 1'b1;
                end
            end
            default: begin
                read_en <= read_en;
                tx_data <= tx_data;
                tx_data_valid <= tx_data_valid;
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
