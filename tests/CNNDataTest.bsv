import cnn::*;
import Config::*;

module mkTb();
  Reg#(int) i <- mkReg(0);
  Reg#(int) j <- mkReg(0);

  let data = cnn_conv2_weight();

  rule print;
    int ni = i;
    int nj = j + 1;
    if (nj == 72) begin
      ni = i + 1;
      nj = 0;
    end
    i <= ni;
    j <= nj;
    $display("i = %d, j = %d, data = %d", i, j, data[i][j]);

    if (ni == 4) begin
      $finish;
    end
  endrule
endmodule