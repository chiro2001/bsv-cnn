import FIFOF::*;
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
    PrimSelectable#(in, Int#(8)),
    PrimSelectable#(out, Int#(8)),
    PrimWriteable#(Reg#(out), Int#(8))
    // PrimWriteable#(out, Int#(8))
    // Bits#(Vector::Vector#(lines, Reg#(Int#(8))), lines_bits)
    // Bits#(Reg#(Int#(8)), 8)
  );
  LayerData_ifc#(Int#(8), lines, depth) data <- mkLayerData(layer_name);
  Reg#(Bool) stage <- mkReg(False);
  Reg#(Bool) done <- mkReg(False);
  // Vector#(lines, Reg#(Int#(8))) tmp <- replicateM(mkReg(0));
  // Vector#(lines, Int#(8)) tmp <- mkReg(unpack('0));
  Reg#(out) tmp <- mkReg(unpack('0));

  FIFOF#(in) fifo_in <- mkFIFOF1;
  FIFOF#(out) fifo_out <- mkFIFOF1;

  rule start (fifo_in.notEmpty && !stage && !done && data.weightsDone() && data.biasDone());
    $display("Layer %s start", layer_name);
    data.resetState();
    // for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
    //   tmp[i] <= 0;
    // end
    tmp <= unpack('0);
  endrule

  rule acc_weights (!stage && !done && !data.weightsDone());
    $display("Layer %s acc weights, index=%x", layer_name, data.getIndex());
    let weight = data.getWeight();
    let top = fifo_in.first;
    out t = tmp;
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      let w <- weight[i];
      let mul = top[i] * w;
      // tmp[i] <= tmp[i] + (mul >> 6);
      t[i] = tmp[i] + (mul >> 6);
    end
    tmp <= t;
    if (data.weightsWillDone()) begin
      stage <= True;
    end
  endrule

  rule acc_weights_to_acc_bias (stage && !done && data.weightsDone() && data.biasDone());
    data.resetState();
  endrule

  rule acc_bias (stage && !done && !data.biasDone());
    $display("Layer %s acc bias, index_bias=%x", layer_name, data.getIndexLines());
    let bias <- data.getBias();
    // tmp[data.getIndexLines()] <= tmp[data.getIndexLines()] + bias;
    out t = tmp;
    t[data.getIndexLines()] = tmp[data.getIndexLines()] + bias;
    tmp <= t;
    if (data.biasWillDone()) begin
      done <= True;
    end
  endrule

  rule acc_bias_to_fifo_out (stage && done && data.biasDone());
    $display("Layer %s acc bias to fifo out", layer_name);
    // fifo_out.enq(unpack(pack(tmp)));
    fifo_out.enq(tmp);
    fifo_in.deq;
    // fifo_out.enq(tmp);
    done <= False;
    stage <= False;
  endrule

  method Action put(in x);
    fifo_in.enq(x);
  endmethod

  method ActionValue#(out) get;
    out y = fifo_out.first;
    fifo_out.deq;
    // fifo_in.deq;
    return y;
  endmethod
endmodule