import FixedPoint::*;

module mkTb();

  FixedPoint#(8, 8) a = 4;
  FixedPoint#(8, 8) b = 3.5;

  rule test_mult;
    let res = a * b;
    $display("a * b = ", fshow(res));
    $finish;
  endrule
endmodule
