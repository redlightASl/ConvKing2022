`timescale 1ns/1ps
`include "i2c_master_defines.v"

module i2c_master_byte_ctrl 
(
clk,rst,nReset,ena,
clk_cnt,start,stop, 
read, write, ack_in,
din,cmd_ack,ack_out,
dout,i2c_busy,i2c_al,
scl_i,scl_o,scl_oen, 
sda_i,sda_o, sda_oen 
);

//
// inputs & outputs
//
input clk;     // master clock
input rst;     // synchronous active high reset(Global reset signal)
input nReset;  // asynchronous active low reset
input ena;     // core enable signal

input [15:0]clk_cnt; // 4x SCL

// control inputs
input       start;
input       stop;
input       read;
input       write;
input       ack_in;
input [7:0] din;

// status outputs
output reg  cmd_ack;
//reg cmd_ack;
output reg  ack_out;
//reg ack_out;
output      i2c_busy;
output      i2c_al;
output [7:0]dout;

// I2C signals
input  scl_i;
output scl_o;
output scl_oen;        
input  sda_i;
output sda_o;
output sda_oen;


// Variable declarations

// statemachine
parameter [4:0] ST_IDLE  = 5'b0_0000;
parameter [4:0] ST_START = 5'b0_0001;
parameter [4:0] ST_READ  = 5'b0_0010;
parameter [4:0] ST_WRITE = 5'b0_0100;
parameter [4:0] ST_ACK   = 5'b0_1000;
parameter [4:0] ST_STOP  = 5'b1_0000;

// signals for bit_controller
reg [3:0]core_cmd;
reg      core_txd;
wire     core_ack, core_rxd;

// signals for shift register
reg [7:0]sr;       //8bit shift register
reg      shift, ld;

// signals for state machine
wire     go;
reg [2:0]dcnt;
wire     cnt_done;

// Module body

// hookup bit_controller
i2c_master_bit_ctrl bit_controller 
(
   .clk      ( clk      ),
    .rst     ( rst      ),
    .nReset  ( nReset   ),
    .ena     ( ena      ),
    .clk_cnt ( clk_cnt  ),
    .cmd     ( core_cmd ),
    .cmd_ack ( core_ack ),
    .busy    ( i2c_busy ),
    .al      ( i2c_al   ),
    .din     ( core_txd ),
    .dout    ( core_rxd ),
    .scl_i   ( scl_i    ),
    .scl_o   ( scl_o    ),
    .scl_oen ( scl_oen  ),
    .sda_i   ( sda_i    ),
    .sda_o   ( sda_o    ),
    .sda_oen ( sda_oen  )
);

// generate go-signal
assign go = (read | write | stop) & ~cmd_ack;   //the “cmd_ack��? was low actived


// generate shift register
always @(posedge clk or negedge nReset)
begin
  if (!nReset)
    sr <= #1 8'h0;                              //the"#"just for modelsim,it's not used when Analysis & Synthesis
  else if (rst==1)
    sr <= #1 8'h0;
  else if (ld)                                  //load data
    sr <= #1 din;                               //din is the input data
  else if (shift)                               
    sr <= #1 {sr[6:0], core_rxd};
end
    
// assign dout output to shift-register
assign dout = sr;

// generate counter
always @(posedge clk or negedge nReset)
begin
  if (!nReset)
    dcnt <= #1 3'h0;
  else if (rst)
    dcnt <= #1 3'h0;
  else if (ld)                                 
    dcnt <= #1 3'h7;
  else if (shift)
    dcnt <= #1 dcnt - 3'h1;
end

assign cnt_done = ~(|dcnt);                     //“|”is Abbreviation operator，just one operand,Operation is performed by bitwise operation from left to right

// state machine
reg [4:0] c_state; // synopsys enum_state

always @(posedge clk or negedge nReset)
if(!nReset)                   
begin
   core_cmd <= #1 `I2C_CMD_NOP;
   core_txd <= #1 1'b0;
   shift    <= #1 1'b0;
   ld       <= #1 1'b0;
   cmd_ack  <= #1 1'b0;
   c_state  <= #1 ST_IDLE;
   ack_out  <= #1 1'b0;
end
else if (rst | i2c_al)
begin
       core_cmd <= #1 `I2C_CMD_NOP;
       core_txd <= #1 1'b0;
       shift    <= #1 1'b0;
       ld       <= #1 1'b0;
       cmd_ack  <= #1 1'b0;
       c_state  <= #1 ST_IDLE;
       ack_out  <= #1 1'b0;
end
else
begin
      // initially reset all signals
      core_txd <= #1 sr[7];
      shift    <= #1 1'b0;
      ld       <= #1 1'b0;
      cmd_ack  <= #1 1'b0;

      case (c_state) // synopsys full_case parallel_case
        ST_IDLE:
        if (go)
        begin
           if (start)
           begin
              c_state  <= #1 ST_START;
              core_cmd <= #1 `I2C_CMD_START;
           end
           else if (read)
           begin
              c_state  <= #1 ST_READ;
              core_cmd <= #1 `I2C_CMD_READ;
           end
           else if (write)
           begin
              c_state  <= #1 ST_WRITE;
              core_cmd <= #1 `I2C_CMD_WRITE;
           end
           else // stop
           begin
              c_state  <= #1 ST_STOP;
              core_cmd <= #1 `I2C_CMD_STOP;
           end
              ld <= #1 1'b1;
        end

        ST_START:
        if (core_ack)
        begin
           if (read)
           begin
              c_state  <= #1 ST_READ;
              core_cmd <= #1 `I2C_CMD_READ;
           end
           else
           begin
              c_state  <= #1 ST_WRITE;
              core_cmd <= #1 `I2C_CMD_WRITE;
           end

              ld <= #1 1'b1;
        end

        ST_WRITE:
        if (core_ack)
           if (cnt_done)
           begin
              c_state  <= #1 ST_ACK;
              core_cmd <= #1 `I2C_CMD_READ;
           end
           else
           begin
              c_state  <= #1 ST_WRITE;       // stay in same state
              core_cmd <= #1 `I2C_CMD_WRITE; // write next bit
              shift    <= #1 1'b1;         
           end

        ST_READ:
        if (core_ack)
        begin
           if (cnt_done)
           begin
              c_state  <= #1 ST_ACK;
              core_cmd <= #1 `I2C_CMD_WRITE;
           end
           else
           begin
              c_state  <= #1 ST_READ;       // stay in same state
              core_cmd <= #1 `I2C_CMD_READ; // read next bit
           end

              shift    <= #1 1'b1;
              core_txd <= #1 ack_in;
        end

        ST_ACK:
        if (core_ack)
        begin
           if (stop)
           begin
              c_state  <= #1 ST_STOP;
              core_cmd <= #1 `I2C_CMD_STOP;
           end
           else
           begin
              c_state  <= #1 ST_IDLE;
              core_cmd <= #1 `I2C_CMD_NOP;

             //generate command acknowledge signal
              cmd_ack  <= #1 1'b1;
           end

              //assign ack_out output to bit_controller(contains last received bit)
              ack_out <= #1 core_rxd;              

              core_txd <= #1 1'b1;
        end
        else
          core_txd <= #1 ack_in;

        ST_STOP:
        if (core_ack)
        begin
           c_state  <= #1 ST_IDLE;
           core_cmd <= #1 `I2C_CMD_NOP;

           // generate command acknowledge signal
           cmd_ack  <= #1 1'b1;
        end

      endcase
  end
endmodule
