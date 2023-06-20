import Vector::*;

module mkTb();

Reg#(Vector#(16, int)) vec_in_reg <- mkReg(replicate(1));

rule test;
  Vector#(16, int) v = vec_in_reg;
  for (Integer i = 0; i < 16; i = i + 1) begin
    v[i] = fromInteger(i + 1) + vec_in_reg[i];
  end
  vec_in_reg <= v;
  let bits = pack(v);
  // let slice = unpack(v)[3:0];
endrule

endmodule