`timescale 1ns/1ps

//Key FSM
module key_top #(
           parameter DELAY_TIME = 20'd1_000_000
       )
       (
           input sys_clk_i,
           input sys_rst_n_i,

           input [1: 0] key_button_i,
           output reg [3: 0] led_indicate_o,

           output reg is_count_mode_o,
           output reg is_color_mode_o,
           output reg is_detect_mode_o
       );
localparam IDLE =           5'b00001;
localparam WAIT =           5'b00010;
localparam COUNT_MODE =     5'b00100;
localparam COLOR_MODE =     5'b01000;
localparam DETECT_MODE =    5'b10000;

wire [1: 0] key_value;

reg [4: 0] current_state;
reg [4: 0] next_state;

//state roll
always @(posedge sys_clk_i or negedge sys_rst_n_i) begin
    if (!sys_rst_n_i)
        current_state <= IDLE;
    else begin
        current_state <= next_state;
    end
end

//condition change
always @( * ) begin
    case (current_state)
        IDLE: begin
            next_state = WAIT;
        end
        WAIT: begin //wait mode,show the basic ov2560 output
            if (key_value[1:0] == 2'b10)
                next_state = COUNT_MODE;
            else if (key_value[1:0] == 2'b01)
                next_state = DETECT_MODE;
            else
                next_state = WAIT;
        end
        COUNT_MODE: begin
            if (key_value[1:0] == 2'b10)
                next_state = COLOR_MODE;
            else if (key_value[1:0] == 2'b01)
                next_state = WAIT;
            else
                next_state = COUNT_MODE;
        end
        COLOR_MODE: begin
            if (key_value[1:0] == 2'b10)
                next_state = DETECT_MODE;
            else if (key_value[1:0] == 2'b01)
                next_state = COUNT_MODE;
            else
                next_state = COLOR_MODE;
        end
        DETECT_MODE: begin
            if (key_value[1:0] == 2'b10)
                next_state = WAIT;
            else if (key_value[1:0] == 2'b01)
                next_state = COLOR_MODE;
            else
                next_state = DETECT_MODE;
        end
        default: begin
            next_state = IDLE;
        end
    endcase
end

//out control
//is_count_mode_o
always @(posedge sys_clk_i or negedge sys_rst_n_i) begin
    if(sys_rst_n_i == 1'b0) 
        is_count_mode_o <= 1'b0;
    else begin
        if(current_state == COUNT_MODE)
            is_count_mode_o <= 1'b1;
        else 
            is_count_mode_o <= 1'b0;
    end
end

//is_color_mode_o
always @(posedge sys_clk_i or negedge sys_rst_n_i) begin
    if(sys_rst_n_i == 1'b0) 
        is_color_mode_o <= 1'b0;
    else begin
        if(current_state == COLOR_MODE)
            is_color_mode_o <= 1'b1;
        else 
            is_color_mode_o <= 1'b0;
    end
end

//is_detect_mode_o
always @(posedge sys_clk_i or negedge sys_rst_n_i) begin
    if(sys_rst_n_i == 1'b0) 
        is_detect_mode_o <= 1'b0;
    else begin
        if(current_state == DETECT_MODE)
            is_detect_mode_o <= 1'b1;
        else 
            is_detect_mode_o <= 1'b0;
    end
end

//led_indicate_o
always @(posedge sys_clk_i or negedge sys_rst_n_i) begin
    if(sys_rst_n_i == 1'b0) 
        led_indicate_o[2:0] <= 3'b111;
    else begin
        if(current_state == WAIT)
            led_indicate_o[2: 0] <= 3'b111;
        else if(current_state == COUNT_MODE)
            led_indicate_o[2: 0] <= 3'b110;
        else if(current_state == COLOR_MODE)
            led_indicate_o[2: 0] <= 3'b101;
        else if(current_state == DETECT_MODE)
            led_indicate_o[2: 0] <= 3'b011;
        else
            led_indicate_o[2: 0] <= led_indicate_o[2: 0];
    end
end

key #(
        .DELAY_TIME ( DELAY_TIME )
    )
    u_key_0(
        //ports
        .sys_clk_i ( sys_clk_i ),
        .sys_rst_n_i ( sys_rst_n_i ),
        .key_button_i ( key_button_i[0] ),
        .key_value_o (),
        .is_change_o ( ),
        .key_posedge_sig_o ( ),
        .key_negedge_sig_o (key_value[0])
    );

key #(
        .DELAY_TIME ( DELAY_TIME )
    )
    u_key_1(
        //ports
        .sys_clk_i ( sys_clk_i ),
        .sys_rst_n_i ( sys_rst_n_i ),
        .key_button_i ( key_button_i[1] ),
        .key_value_o ( ),
        .is_change_o ( ),
        .key_posedge_sig_o ( ),
        .key_negedge_sig_o (key_value[1] )
    );
endmodule
