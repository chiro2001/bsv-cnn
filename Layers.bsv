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
  );
  LayerData_ifc#(Int#(8), lines, depth) data <- mkLayerData(layer_name);
  Reg#(Bool) done <- mkReg(True);
  Reg#(out) tmp <- mkReg(unpack('0));

  FIFOF#(in) fifo_in <- mkFIFOF1;
  FIFOF#(out) fifo_out <- mkFIFOF1;

  rule start (done && fifo_in.notEmpty && data.weightsDone() && data.biasDone());
    // $display("Layer %s start", layer_name);
    data.weightStart();
    data.biasStart();
    done <= False;
    tmp <= unpack('0);
  endrule

  rule acc_weights_bias (!done && !data.weightsDone() && !data.biasDone());
    let index = data.getIndex() - 1;
    let index_bias = data.getIndexLines() - 1;
    // $display("Layer %s acc weights, index=%x", layer_name, index);
    let weight = data.getWeight();
    let top = fifo_in.first;
    out t = tmp;
    let bias <- data.getBias();
    // if (layer_name == "fc2") $display("Layer %s: bias=%x", layer_name, bias);
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      let w <- weight[i];
      let mul = top[i] * w;
      if (fromInteger(i) == index_bias)
        t[i] = tmp[i] + (mul >> 6) + bias;
      else t[i] = tmp[i] + (mul >> 6);
    end
    tmp <= t;
    // if (layer_name == "fc2") $display("Layer %s: t=%x", layer_name, pack(t));
  endrule

  // rule display_index (fifo_in.notEmpty && layer_name == "fc2");
  //   let index = data.getIndex() - 1;
  //   let index_bias = data.getIndexLines() - 1;
  //   $display("Layer %s: done=%x, index=%x, index_bias=%x, weights_done=%x, bias_done=%x, tmp=%x", 
  //     layer_name, done, index, index_bias, data.weightsDone(), data.biasDone(), pack(tmp));
  // endrule

  rule acc_weights_only (!done && !data.weightsDone() && data.biasDone());
    let index = data.getIndex() - 1;
    // $display("Layer %s acc weights only, index=%x", layer_name, index);
    let weight = data.getWeight();
    let top = fifo_in.first;
    out t = tmp;
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      let w <- weight[i];
      let mul = top[i] * w;
      t[i] = tmp[i] + (mul >> 6);
    end
    tmp <= t;
  endrule

  rule acc_bias_only (!done && data.weightsDone() && !data.biasDone());
    let index_bias = data.getIndexLines() - 1;
    // $display("Layer %s acc bias only, index_bias=%x", layer_name, index_bias);
    let bias <- data.getBias();
    out t = tmp;
    t[index_bias] = tmp[index_bias] + bias;
    tmp <= t;
  endrule

  rule set_done (!done && data.weightsDone() && data.biasDone());
    // $display("Layer %s set done, tmp=%x", layer_name, pack(tmp));
    done <= True;
    fifo_out.enq(tmp);
    fifo_in.deq;
    tmp <= unpack('0);
  endrule

  method Action put(in x);
    fifo_in.enq(x);
  endmethod

  method ActionValue#(out) get;
    out y = fifo_out.first;
    fifo_out.deq;
    return y;
  endmethod
endmodule