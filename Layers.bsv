import FIFOF::*;
import FIFO::*;
import Vector::*;
import Data::*;
import Utils::*;

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
    Mul#(lines, 16, lines_bits), 
    Mul#(depth, 16, depth_bits),
    PrimSelectable#(in, Int#(16)),
    PrimSelectable#(out, Int#(16)),
    PrimWriteable#(Reg#(out), Int#(16))
  );
  LayerData_ifc#(Int#(16), lines, depth) data <- mkLayerData("fc", layer_name);
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
    Vector#(lines, Int#(16)) weight = unpack(weight_data);
    let top = fifo_in.first;
    out t = tmp;
    let bias <- data.getBias();
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      let w = weight[i];
      let mul = top[i] * w;
      if (fromInteger(i) == index_bias)
        t[i] = tmp[i] + (mul>> q_bits()) + bias;
      else t[i] = tmp[i] + (mul>> q_bits());
    end
    tmp <= t;
  endrule

  rule acc_weights_only (!done && !data.weightsDone() && data.biasDone());
    let index = data.getWeightsIndex() - 1;
    // $display("Layer %s acc weights only, index=%x", layer_name, index);
    let weight_data <- data.getWeights();
    Vector#(lines, Int#(16)) weight = unpack(weight_data);
    let top = fifo_in.first;
    out t = tmp;
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      let w = weight[i];
      let mul = top[i] * w;
      t[i] = tmp[i] + (mul>> q_bits());
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
    Mul#(input_size, 16, input_bits), 
    PrimSelectable#(in, Int#(16)),
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
    Mul#(input_size, 16, input_bits), 
    PrimSelectable#(in, Int#(16)),
    Bits#(out, output_bits), 
    Mul#(output_size, 16, output_bits), 
    PrimSelectable#(out, Int#(16)),
    Add#(input_bits, 0, output_bits),
    PrimUpdateable#(out, Int#(16))
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
    Mul#(input_size, 16, input_bits),
    Mul#(input_lines, input_lines, input_size),
    Bits#(out, output_bits),
    Mul#(TMul#(output_size, 16), output_channels, output_bits),
    Mul#(output_lines, output_lines, output_size),
    // 2D vectors required
    PrimSelectable#(in, Vector::Vector#(input_lines, Int#(16))),
    PrimSelectable#(out, Vector::Vector#(output_lines, Vector::Vector#(output_lines, Int#(16)))),
    Add#(output_lines, kernel_size, TAdd#(input_lines, 1)),
    Mul#(kernel_size, kernel_size, kernel_size_2),
    Mul#(kernel_size_2, 16, kernel_size_2_bits),
    Add#(kernel_size, 0, 3),
    PrimUpdateable#(out, Vector::Vector#(output_lines, Vector::Vector#(output_lines, Int#(16))))
  );

  FIFOF#(in) fifo_in <- mkFIFOF1;
  FIFOF#(out) fifo_out <- mkFIFOF1;

  Wire#(in) data_in <- mkWire;

  rule set_data_in;
    data_in <= fifo_in.first;
  endrule

  Wire#(Vector#(output_lines, Vector#(output_lines, Bit#(kernel_size_2_bits)))) cols <- mkWire;
  // Vector#(output_lines, Vector#(output_lines, Reg#(Int#(16)))) data_out <- replicateM(replicateM(mkReg(0)));
  // Reg#(Vector#(output_lines, Vector#(output_lines, Int#(16)))) data_out <- mkReg(unpack('0));

  rule bind_cols;
    Vector#(output_lines, Vector#(output_lines, Bit#(kernel_size_2_bits))) cols_ = unpack('0);
    for (Integer i = 0; i < valueOf(output_lines); i = i + 1) begin
      for (Integer j = 0; j < valueOf(output_lines); j = j + 1) begin
        cols_[i][j] = {
          pack(data_in[i][j]),
          pack(data_in[i][j + 1]),
          pack(data_in[i][j + 2]),
          pack(data_in[i + 1][j]),
          pack(data_in[i + 1][j + 1]),
          pack(data_in[i + 1][j + 2]),
          pack(data_in[i + 2][j]),
          pack(data_in[i + 2][j + 1]),
          pack(data_in[i + 2][j + 2])
        };
      end
    end
    cols <= cols_;
  endrule

  LayerData_ifc#(Int#(16), kernel_size_2, output_channels) data <- mkLayerData("cnn", layer_name);
  Reg#(Bool) done <- mkReg(True);
  Reg#(out) tmp <- mkReg(unpack('0));

  rule start (done && fifo_in.notEmpty && data.weightsDone() && data.biasDone());
    // $display("Layer %s start", layer_name);
    data.weightsStart();
    data.biasStart();
    done <= False;
    tmp <= unpack('0);
  endrule

  rule acc (!done && !data.weightsDone() && !data.biasDone());
    let index = data.getWeightsIndex() - 1;
    // $display("Layer %s acc weights, index=%x", layer_name, index);
    let weight_data <- data.getWeights();
    Vector#(kernel_size_2, Int#(16)) kernel = unpack(weight_data);
    let top = fifo_in.first;
    out t = tmp;
    let bias <- data.getBias();
    for (Integer i = 0; i < valueOf(output_lines); i = i + 1) begin
      for (Integer j = 0; j < valueOf(output_lines); j = j + 1) begin
        Int#(16) s = 0;
        Vector#(kernel_size_2, Int#(16)) col = unpack(cols[i][j]);
        // for (Integer k = 0; k < valueOf(kernel_size_2); k = k + 1) begin
        //   s = s + ((col[k] * kernel[k])>> q_bits());
        // end
        s = ((col[0] * kernel[0])>> q_bits());
        t[index][i][j] = s + bias;
      end
    end
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