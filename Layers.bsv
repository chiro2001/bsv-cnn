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
    Add#(TLog#(lines), v__, TLog#(depth))
  );
  LayerData_ifc#(Int#(8), lines, depth) data <- mkLayerData(layer_name);

  FIFO#(in) fifo_in <- mkLFIFO;
  FIFO#(out) fifo_out <- mkLFIFO;

  method Action put(in x);
    fifo_in.enq(x);
  endmethod

  method ActionValue#(out) get;
    out y = fifo_out.first;
    fifo_in.deq;
    return y;
  endmethod
endmodule