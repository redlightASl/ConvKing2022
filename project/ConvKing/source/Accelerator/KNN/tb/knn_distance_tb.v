module knn_distance_tb();

reg [15:0]  img;
reg [15:0]  dic;
wire [13:0] distance;

initial begin
    img=16'b1000_0110_0111_1101;
    dic=16'b10000_100000_10000;
end

knn_distance inst(
    .img    (img),
    .dic    (dic),
    .distance   (distance)
);


endmodule