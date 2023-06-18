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

  rule start if (!stage && data.getIndex() == 0);
    fifo_in.deq;
    data.resetState();
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      tmp[i] <= 0;
    end
  endrule

  rule acc_weights if (!stage && data.getIndex() < fromInteger(valueOf(depth)));
    let weight = data.getWeight();
    let top = fifo_in.first;
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      let w <- weight[i];
      tmp[i] <= tmp[i] + (top[i] * w) >> 6;
    end
    if (data.getIndex() == fromInteger(valueOf(depth) - 1))
      stage <= True;
  endrule

  rule acc_bias if (stage && data.getIndexLines() < fromInteger(valueOf(lines)));
    if (data.getIndexLines() == fromInteger(valueOf(lines) - 1))
      stage <= False;
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