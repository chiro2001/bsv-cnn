import FIFO::*;
import Vector::*;
import Data::*;

interface Layer#(type in, type out);
  method Action put(in x);
  method ActionValue#(out) get;
endinterface

module mkFCLayer#(parameter String layer_name)(Layer#(in, out))
  provisos(
    Bits#(out, lines_bits), 
    Bits#(in, depth_bits), 
    Mul#(lines, 8, lines_bits), 
    Mul#(depth, 8, depth_bits),
    PrimSelectable#(in, Int#(8))
  );
  LayerData_ifc#(Int#(8), lines, depth) data <- mkLayerData(layer_name);
  Reg#(Bool) stage <- mkReg(False);
  Vector#(lines, Reg#(Int#(8))) tmp <- replicateM(mkReg(0));

  FIFO#(in) fifo_in <- mkLFIFO;
  FIFO#(out) fifo_out <- mkLFIFO;

  Reg#(in) data_in <- mkReg(unpack('0));

  rule start (!stage && data.weightsDone() && data.biasDone());
    data_in <= fifo_in.first;
    data.resetState();
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      tmp[i] <= 0;
    end
  endrule

  rule acc_weights (!stage && !data.weightsDone());
    let weight = data.getWeight();
    let top = data_in;
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      let w <- weight[i];
      let mul = top[i] * w;
      tmp[i] <= tmp[i] + mul >> 6;
    end
    if (data.weightsWillDone()) stage <= True;
  endrule

  rule acc_bias (stage && !data.biasDone());
    if (data.biasWillDone()) stage <= False;
  endrule

  method Action put(in x);
    fifo_in.enq(x);
  endmethod

  method ActionValue#(out) get;
    out y = fifo_out.first;
    fifo_in.deq;
    return y;
  endmethod
endmodule