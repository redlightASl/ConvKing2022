module hsv2rgb(
           input clk,
           input reset_n,
           input [8: 0]i_hsv_h,
           input [8: 0]i_hsv_s,
           input [7: 0]i_hsv_v,
           input vs,
           input hs,
           input de,
           output [7: 0]rgb_r,
           output [7: 0]rgb_g,
           output [7: 0]rgb_b,
           output rgb_vs,
           output rgb_hs,
           output rgb_de
       );


reg [7: 0]i_hsv_v_r1;
reg [7: 0]i_hsv_v_r2;
reg [7: 0]i_hsv_v_r3;
reg [7: 0]i_hsv_v_r4;
reg [8: 0]i_hsv_h_r;
reg [8: 0]i_hsv_s_r1;
reg [8: 0]i_hsv_s_r2;
reg [8: 0]i_hsv_s_r3;
reg [8: 0]i_hsv_s_r4;

wire [8: 0]temp;
assign temp = 9'd256 - i_hsv_s;

reg [16: 0] p; //p=V-V*S,，扩大256
reg [7: 0] p2;
reg [7: 0]p3;
reg [2: 0]I; //H/60的商
reg [2: 0]I2;
reg [2: 0]I3;
wire [5: 0]f; //H/60的余数
assign f = i_hsv_h_r - I * 60;
reg [15: 0]f_60;

reg [15: 0]adjust;



always@(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        i_hsv_v_r1 <= 0;
        i_hsv_v_r2 <= 0;
        i_hsv_v_r3 <= 0;
        i_hsv_v_r4 <= 0;
    end
    else begin
        i_hsv_v_r1 <= i_hsv_v;
        i_hsv_v_r2 <= i_hsv_v_r1;
        i_hsv_v_r3 <= i_hsv_v_r2;
        i_hsv_v_r4 <= i_hsv_v_r3;
    end
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        i_hsv_h_r <= 0;
    else
        i_hsv_h_r <= i_hsv_h;
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        i_hsv_s_r1 <= 0;
        i_hsv_s_r2 <= 0;
        i_hsv_s_r3 <= 0;
        i_hsv_s_r4 <= 0;
    end
    else begin
        i_hsv_s_r1 <= i_hsv_s;
        i_hsv_s_r2 <= i_hsv_s_r1;
        i_hsv_s_r3 <= i_hsv_s_r2;
        i_hsv_s_r4 <= i_hsv_s_r3;
    end
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        p <= 0;
        p2 <= 0;
        p3 <= 0;
    end
    else begin
        p <= temp * i_hsv_v;
        p2 <= p[15: 8];
        p3 <= p2;
    end
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        I <= 0;
    else if (i_hsv_h < 60)
        I <= 0;
    else if ((i_hsv_h < 120) && (i_hsv_h >= 60))
        I <= 1;
    else if ((i_hsv_h < 180) && (i_hsv_h >= 120))
        I <= 2;
    else if ((i_hsv_h < 240) && (i_hsv_h >= 180))
        I <= 3;
    else if ((i_hsv_h < 300) && (i_hsv_h >= 240))
        I <= 4;
    else
        I <= 5;
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        I2 <= 0;
        I3 <= 0;
    end
    else begin
        I2 <= I;
        I3 <= I2;
    end
end


always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        f_60 <= 0;
    else
        f_60 <= {f, 8'b0} / 60;
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        adjust <= 0;
    else
        adjust <= (i_hsv_v_r2 - p2) * f_60[7: 0];
end


/**************************************/
reg [7: 0]rgb_r_r;
reg [7: 0]rgb_g_r;
reg [7: 0]rgb_b_r;

reg [3: 0]vs_delay;
reg [3: 0]hs_delay;
reg [3: 0]de_delay;

always@(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        rgb_r_r <= 0;
        rgb_g_r <= 0;
        rgb_b_r <= 0;
    end
    else
    case (I3)
        0: begin
            rgb_r_r <= i_hsv_v_r3;
            rgb_g_r <= p3 + adjust[15: 8];
            rgb_b_r <= p3;
        end

        1: begin
            rgb_r_r <= i_hsv_v_r3 - adjust[15: 8];
            rgb_g_r <= i_hsv_v_r3;
            rgb_b_r <= p3;
        end

        2: begin
            rgb_r_r <= p3;
            rgb_g_r <= i_hsv_v_r3;
            rgb_b_r <= p3 + adjust[15: 8];
        end

        3: begin
            rgb_r_r <= p3;
            rgb_g_r <= i_hsv_v_r3 - adjust[15: 8];
            rgb_b_r <= i_hsv_v_r3;
        end

        4: begin
            rgb_r_r <= p3 + adjust[15: 8];
            rgb_g_r <= p3;
            rgb_b_r <= i_hsv_v_r3;
        end

        5: begin
            rgb_r_r <= i_hsv_v_r3;
            rgb_g_r <= p3;
            rgb_b_r <= i_hsv_v_r3 - adjust[15: 8];
        end

        default: begin
            rgb_r_r <= 0;
            rgb_g_r <= 0;
            rgb_b_r <= 0;
        end
    endcase
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        vs_delay <= 0;
        hs_delay <= 0;
        de_delay <= 0;
    end
    else begin
        vs_delay <= {vs_delay[2: 0], vs};
        hs_delay <= {hs_delay[2: 0], hs};
        de_delay <= {de_delay[2: 0], de};
    end
end

assign rgb_r = (i_hsv_s_r4 == 0) ? i_hsv_v_r4 : rgb_r_r;
assign rgb_g = (i_hsv_s_r4 == 0) ? i_hsv_v_r4 : rgb_g_r;
assign rgb_b = (i_hsv_s_r4 == 0) ? i_hsv_v_r4 : rgb_b_r;

assign rgb_vs = vs_delay[3];
assign rgb_hs = hs_delay[3];
assign rgb_de = de_delay[3];

endmodule
