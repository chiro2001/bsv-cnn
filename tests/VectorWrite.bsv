import Vector::*;

module mkTb();

Reg#(Vector#(8, int)) vec_in_reg <- mkReg(replicate(1));

rule test;
  Vector#(8, int) v = vec_in_reg;
  for (Integer i = 0; i < 8; i = i + 1) begin
    v[i] = fromInteger(i + 1) + vec_in_reg[i];
  end
  vec_in_reg <= v;
  let bits = pack(v);
  // let slice = unpack(v)[3:0];
endrule

endmodule