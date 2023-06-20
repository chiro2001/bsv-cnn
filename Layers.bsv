import FIFOF::*;
import FIFO::*;
import Vector::*;
import Data::*;

interface Layer#(type in, type out);
  method Action put(in x);
  method ActionValue#(out) get;
endinterface

interface LayerDirect#(type in, type out);
  method Action put(in x);
  method out get;
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
  LayerData_ifc#(Int#(8), lines, depth) data <- mkLayerData("fc", layer_name);
  Reg#(Bool) done <- mkReg(True);
  Reg#(out) tmp <- mkReg(unpack('0));

  FIFOF#(in) fifo_in <- mkFIFOF1;
  FIFOF#(out) fifo_out <- mkFIFOF1;

  rule start (done && fifo_in.notEmpty && data.weightsDone() && data.biasDone());
    // $display("Layer %s start", layer_name);
    data.weightsStart();
    data.biasStart();
    done <= False;
    tmp <= unpack('0);
  endrule

  rule acc_weights_bias (!done && !data.weightsDone() && !data.biasDone());
    let index = data.getWeightsIndex() - 1;
    let index_bias = data.getBiasIndex() - 1;
    // $display("Layer %s acc weights, index=%x", layer_name, index);
    let weight_data <- data.getWeights();
    Vector#(lines, Int#(8)) weight = unpack(weight_data);
    let top = fifo_in.first;
    out t = tmp;
    let bias <- data.getBias();
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      let w = weight[i];
      let mul = top[i] * w;
      if (fromInteger(i) == index_bias)
        t[i] = tmp[i] + (mul >> 6) + bias;
      else t[i] = tmp[i] + (mul >> 6);
    end
    tmp <= t;
  endrule

  rule acc_weights_only (!done && !data.weightsDone() && data.biasDone());
    let index = data.getWeightsIndex() - 1;
    // $display("Layer %s acc weights only, index=%x", layer_name, index);
    let weight_data <- data.getWeights();
    Vector#(lines, Int#(8)) weight = unpack(weight_data);
    let top = fifo_in.first;
    out t = tmp;
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      let w = weight[i];
      let mul = top[i] * w;
      t[i] = tmp[i] + (mul >> 6);
    end
    tmp <= t;
  endrule

  rule acc_bias_only (!done && data.weightsDone() && !data.biasDone());
    let index_bias = data.getBiasIndex() - 1;
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
    fifo_out.deq;
    return fifo_out.first;
  endmethod
endmodule

module mkSoftmaxLayer(Layer#(in, out))
  provisos (
    Bits#(in, input_bits), 
    Mul#(input_size, 8, input_bits), 
    PrimSelectable#(in, Int#(8)),
    Bits#(out, output_bits),
    PrimIndex#(out, a__),
    Log#(input_size, output_bits)
  );

  FIFO#(out) fifo_out <- mkFIFO1;

  method Action put(in x);
    out y = unpack('0);
    for (Integer i = 0; i < valueOf(input_size); i = i + 1) begin
      if (x[i] > x[y]) y = fromInteger(i);
    end
    fifo_out.enq(y);
  endmethod

  method ActionValue#(out) get;
    fifo_out.deq;
    return fifo_out.first;
  endmethod

endmodule

module mkReluLayer(Layer#(in, out))
  provisos (
    Bits#(in, input_bits), 
    Mul#(input_size, 8, input_bits), 
    PrimSelectable#(in, Int#(8)),
    Bits#(out, output_bits), 
    Mul#(output_size, 8, output_bits), 
    PrimSelectable#(out, Int#(8)),
    Add#(input_bits, 0, output_bits),
    PrimUpdateable#(out, Int#(8))
  );

  FIFO#(out) fifo_out <- mkFIFO1;

  method Action put(in x);
    out y;
    for (Integer i = 0; i < valueOf(input_size); i = i + 1) begin
      if (x[i] < 0) y[i] = 0;
      else y[i] = x[i];
    end
    fifo_out.enq(y);
  endmethod

  method ActionValue#(out) get;
    fifo_out.deq;
    return fifo_out.first;
  endmethod

endmodule

module mkConvLayer#(parameter String layer_name)(Layer#(in, out))
// now assuming that stride == 1
  provisos (
    Bits#(in, input_bits),
    Mul#(input_size, 8, input_bits),
    Mul#(input_lines, input_lines, input_size),
    Bits#(out, output_bits),
    Mul#(output_size, 8, output_bits),
    Mul#(output_lines, output_lines, output_size),
    // 2D vectors required
    PrimSelectable#(in, Vector::Vector#(input_lines, Int#(8))),
    PrimSelectable#(out, Vector::Vector#(output_lines, Int#(8))),
    Add#(output_lines, kernel_size, TAdd#(input_lines, 1)),
    Mul#(kernel_size, kernel_size, kernel_size_2)
  );

  FIFOF#(in) fifo_in <- mkFIFOF1;
  FIFOF#(out) fifo_out <- mkFIFOF1;

  Wire#(in) data_in <- 
    // mkDWire(unpack('0));
    mkWire;

  rule set_data_in;
    data_in <= fifo_in.first;
  endrule

  Wire#(Vector#(output_size, Vector#(kernel_size_2, Int#(8)))) cols <- 
    // mkDWire(unpack('0));
    mkWire;

  rule bind_cols;
    Vector#(output_size, Vector#(kernel_size_2, Int#(8))) cols_ = unpack(0);
    for (Integer i = 0; i < valueOf(output_lines); i = i + 1) begin
      for (Integer j = 0; j < valueOf(output_lines); j = j + 1) begin
        let idx = (i * valueOf(output_lines)) + j;
        for (Integer a = 0; a < valueOf(kernel_size); a = a + 1) begin
          for (Integer b = 0; b < valueOf(kernel_size); b = b + 1) begin
            let k_idx = (a * valueOf(kernel_size)) + b;
            cols_[idx][k_idx] = data_in[i + a][j + b];
          end
        end
      end
    end
    // for (Integer i = 0; i < valueOf(output_size); i = i + 1) begin
    //   for (Integer k = 0; k < valueOf(kernel_size_2); k = k + 1) begin
    //     cols_[i][k] = data_in[i / valueOf(output_lines) + k / valueOf(kernel_size)][i % valueOf(output_lines) + k % valueOf(kernel_size)];
    //   end
    // end
    cols <= cols_;
  endrule

  // rule test;
  //   // $display("input_lines=%d, output_lines=%d, kernel_size=%d", valueOf(input_lines), valueOf(output_lines), valueOf(kernel_size));
  //   for (Integer i = 0; i < valueOf(output_lines); i = i + 1) begin
  //     for (Integer j = 0; j < valueOf(output_lines); j = j + 1) begin
  //       let idx = (i * valueOf(output_lines)) + j;
  //       $display("cols[i=%d][j=%d] %x", cols[idx][0]);
  //     end
  //   end
  // endrule

  // LayerData_ifc#(Int#(8), lines, depth) data <- mkLayerData("cnn", layer_name);
  // Reg#(Bool) done <- mkReg(True);
  // Reg#(out) tmp <- mkReg(unpack('0));

  // rule start (done && fifo_in.notEmpty && data.weightsDone() && data.biasDone());
  //   // $display("Layer %s start", layer_name);
  //   data.weightsStart();
  //   data.biasStart();
  //   done <= False;
  //   tmp <= unpack('0);
  // endrule

  // rule acc_weights_bias (!done && !data.weightsDone() && !data.biasDone());
  //   let index = data.getWeightsIndex() - 1;
  //   let index_bias = data.getBiasIndex() - 1;
  //   // $display("Layer %s acc weights, index=%x", layer_name, index);
  //   let weight = data.getWeights();
  //   let top = fifo_in.first;
  //   out t = tmp;
  //   let bias <- data.getBias();
  //   for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
  //     let w <- weight[i];
  //     let mul = top[i] * w;
  //     if (fromInteger(i) == index_bias)
  //       t[i] = tmp[i] + (mul >> 6) + bias;
  //     else t[i] = tmp[i] + (mul >> 6);
  //   end
  //   tmp <= t;
  // endrule

  // rule acc_weights_only (!done && !data.weightsDone() && data.biasDone());
  //   let index = data.getWeightsIndex() - 1;
  //   // $display("Layer %s acc weights only, index=%x", layer_name, index);
  //   let weight = data.getWeights();
  //   let top = fifo_in.first;
  //   out t = tmp;
  //   for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
  //     let w <- weight[i];
  //     let mul = top[i] * w;
  //     t[i] = tmp[i] + (mul >> 6);
  //   end
  //   tmp <= t;
  // endrule

  // rule acc_bias_only (!done && data.weightsDone() && !data.biasDone());
  //   let index_bias = data.getBiasIndex() - 1;
  //   // $display("Layer %s acc bias only, index_bias=%x", layer_name, index_bias);
  //   let bias <- data.getBias();
  //   out t = tmp;
  //   t[index_bias] = tmp[index_bias] + bias;
  //   tmp <= t;
  // endrule

  // rule set_done (!done && data.weightsDone() && data.biasDone());
  //   // $display("Layer %s set done, tmp=%x", layer_name, pack(tmp));
  //   done <= True;
  //   fifo_out.enq(tmp);
  //   fifo_in.deq;
  //   tmp <= unpack('0);
  // endrule

  method Action put(in x);
    fifo_in.enq(x);
  endmethod

  method ActionValue#(out) get;
    out y = fifo_out.first;
    fifo_out.deq;
    return y;
  endmethod

endmodule