module rgb2hsv(
           input clk,
           input reset_n,
           input [7: 0]rgb_r,
           input [7: 0]rgb_g,
           input [7: 0]rgb_b,
           input vs,
           input hs,
           input de,
           output [8: 0]hsv_h,
           output [8: 0]hsv_s,
           output [7: 0]hsv_v,
           output hsv_vs,
           output hsv_hs,
           output hsv_de
       );


reg [7: 0]max;
reg	[7: 0]min;
reg	[13: 0]rgb_r_r;
reg	[13: 0]rgb_g_r;
reg	[13: 0]rgb_b_r;

reg [13: 0]rgb_r_r2;
reg	[13: 0]rgb_g_r2;
reg	[13: 0]rgb_b_r2;
reg	[7: 0]max_r;

wire [7: 0]max_min;
assign	max_min = max - min;
reg [7: 0]max_min_r;
wire [13: 0]max60;
assign max60 = max * 60;

wire [13: 0] g_b;
wire [13: 0] b_r;
wire [13: 0] r_g;
assign	g_b = (rgb_g_r >= rgb_b_r) ? (rgb_g_r - rgb_b_r) : (rgb_b_r - rgb_g_r);
assign b_r = (rgb_b_r >= rgb_r_r) ? (rgb_b_r - rgb_r_r) : (rgb_r_r - rgb_b_r);
assign r_g = (rgb_r_r >= rgb_g_r) ? (rgb_r_r - rgb_g_r) : (rgb_g_r - rgb_r_r);


reg [13: 0]temp;
reg	[13: 0]hsv_h_r;
reg	[15: 0]hsv_s_r;
reg	[7: 0]hsv_v_r;


always@(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        rgb_r_r <= 0;
        rgb_g_r <= 0;
        rgb_b_r <= 0;
    end
    else begin
        rgb_r_r <= 60 * rgb_r;
        rgb_g_r <= 60 * rgb_g;
        rgb_b_r <= 60 * rgb_b;
    end
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        rgb_r_r2 <= 0;
        rgb_g_r2 <= 0;
        rgb_b_r2 <= 0;
    end
    else begin
        rgb_r_r2 <= rgb_r_r;
        rgb_g_r2 <= rgb_g_r;
        rgb_b_r2 <= rgb_b_r;
    end
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        max <= 0;
    else if ((rgb_r >= rgb_b) && (rgb_r >= rgb_g))
        max <= rgb_r;
    else if ((rgb_g >= rgb_b) && (rgb_g >= rgb_r))
        max <= rgb_g;
    else if ((rgb_b >= rgb_r) && (rgb_b >= rgb_g))
        max <= rgb_b;
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        min <= 0;
    else if ((rgb_r <= rgb_b) && (rgb_r <= rgb_g))
        min <= rgb_r;
    else if ((rgb_g <= rgb_b) && (rgb_g <= rgb_r))
        min <= rgb_g;
    else if ((rgb_b <= rgb_r) && (rgb_b <= rgb_g))
        min <= rgb_b;
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        max_min_r <= 0;
    else
        max_min_r <= max_min;
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        temp <= 0;
    else if (max_min != 0) begin
        if (rgb_r_r == max60)
            temp <= g_b / {6'b0, max_min};
        else if (rgb_g_r == max60)
            temp <= b_r / {6'b0, max_min};
        else if (rgb_b_r == max60)
            temp <= r_g / {6'b0, max_min};
    end
    else if (max_min == 0)
        temp <= 0;
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        max_r <= 0;
    else
        max_r <= max;
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        hsv_h_r <= 0;
    else if (max_r == 0)
        hsv_h_r <= 0;
    else if (rgb_r_r2 == 60 * max_r)
        hsv_h_r <= (rgb_g_r2 >= rgb_b_r2) ? temp : (14'd360 - temp);
    else if (rgb_g_r2 == 60 * max_r)
        hsv_h_r <= (rgb_b_r2 >= rgb_r_r2) ? (temp + 120) : (14'd120 - temp);
    else if (rgb_b_r2 == 60 * max_r)
        hsv_h_r <= (rgb_r_r2 >= rgb_g_r2) ? (temp + 240) : (14'd240 - temp);
end

always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        hsv_s_r <= 0;
    else if (max_r == 0)
        hsv_s_r <= 0;
    else
        hsv_s_r <= {max_min_r, 8'b0} / {8'b0, max_r};
end


always@(posedge clk or negedge reset_n) begin
    if (!reset_n)
        hsv_v_r <= 0;
    else
        hsv_v_r <= max_r;
end

reg [2: 0]vs_delay;
reg	[2: 0]hs_delay;
reg	[2: 0]de_delay;


always@(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        vs_delay <= 0;
        hs_delay <= 0;
        de_delay <= 0;
    end
    else begin
        vs_delay <= { vs_delay[1: 0], vs};
        hs_delay <= { hs_delay[1: 0], hs};
        de_delay <= { de_delay[1: 0], de};
    end
end

assign hsv_vs = vs_delay[2];
assign hsv_hs = hs_delay[2];
assign hsv_de = de_delay[2];

assign hsv_h = hsv_h_r[8: 0];
assign hsv_s = hsv_s_r[8: 0];
assign hsv_v = hsv_v_r;

endmodule
