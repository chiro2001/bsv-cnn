import FIFOF::*;
import FIFO::*;
import Vector::*;
import FixedPoint::*;

import Data::*;
import Utils::*;
import Config::*;

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
    Mul#(lines, SizeOf#(ElementType), lines_bits), 
    Mul#(depth, SizeOf#(ElementType), depth_bits),
    PrimSelectable#(in, ElementType),
    PrimSelectable#(out, ElementType),
    PrimWriteable#(Reg#(out), ElementType),
    PrimWriteable#(Reg#(Vector#(lines, ElementTmpType)), ElementTmpType),
    Add#(TLog#(lines), a__, TLog#(depth))
  );
  LayerData_ifc#(ElementType, lines, depth) data <- mkLayerData("fc", layer_name);
  Reg#(Bool) done <- mkReg(True);
  // Reg#(out) tmp <- mkReg(unpack('0));
  Reg#(Vector#(lines, ElementTmpType)) tmp <- mkReg(unpack('0));

  FIFOF#(in) fifo_in <- mkFIFOF;
  FIFOF#(out) fifo_out <- mkFIFOF;

  rule start if (done && fifo_in.notEmpty && data.weightsDone() && data.biasDone());
    data.weightsStart();
    data.biasStart();
    done <= False;
    tmp <= unpack('0);
  endrule

  rule acc_weights_bias if (!done && !data.weightsDone() && !data.biasDone());
    let index = data.getWeightsIndex() - 1;
    // let index_bias = data.getBiasIndex() - 1;
    let weight <- data.getWeights();
    let top = fifo_in.first;
    Vector#(lines, ElementTmpType) t = tmp;
    let bias <- data.getBias();
    ElementTmpType bias_tmp = elementExtend(bias);
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      ElementTmpType mul = elementMult(top[index], weight[i]);
      t[i] = (index == 0 ? 0 : tmp[i]) + mul + (fromInteger(i) == index ? bias_tmp : 0);
    end
    // if (index < 6 && layer_name == "fc1") begin
    //   $display("[idx=%d] t[0] = %d, tmp[0] = %d", index, elementToInt(elementTruncate(t[0])), elementToInt(elementTruncate(tmp[0])));
    // end
    tmp <= t;
    data.weightsInc();
    data.biasInc();
  endrule

  rule acc_weights_only if (!done && !data.weightsDone() && data.biasDone());
    let index = data.getWeightsIndex() - 1;
    let weight <- data.getWeights();
    let top = fifo_in.first;
    Vector#(lines, ElementTmpType) t = tmp;
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      ElementTmpType mul = elementMult(top[index], weight[i]);
      t[i] = (index == 0 ? 0 : tmp[i]) + mul;
    end
    tmp <= t;
    data.weightsInc();
  endrule

  rule set_done if (!done && data.weightsDone() && data.biasDone());
    done <= True;
    // fifo_out.enq(tmp);
    out o;
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      o[i] = elementTruncate(tmp[i]);
    end
    fifo_out.enq(o);
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
    Mul#(input_size, SizeOf#(ElementType), input_bits), 
    PrimSelectable#(in, ElementType),
    Bits#(out, output_bits),
    PrimIndex#(out, a__)
  );

  FIFO#(out) fifo_out <- mkFIFO1;

  method Action put(in x);
    out y = unpack('0);
    // for (Integer i = 0; i < valueOf(input_size); i = i + 1) begin
    //   $write("%d ", elementToInt(x[i]));
    // end
    // $display("");
    // just `hard' max
    for (Integer i = 1; i < valueOf(input_size); i = i + 1) begin
      if (x[i] > x[y]) begin
        // $write("max from %d ", y);
        // $write(fshow(x[y]));
        // $write(" to ");
        // $write(fshow(x[i]));
        // $display(" %d", i);
        y = fromInteger(i);
      end
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
    Mul#(input_size, SizeOf#(ElementType), input_bits), 
    PrimSelectable#(in, ElementType),
    Bits#(out, output_bits), 
    Mul#(output_size, SizeOf#(ElementType), output_bits), 
    PrimSelectable#(out, ElementType),
    Add#(input_bits, 0, output_bits),
    PrimUpdateable#(out, ElementType)
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
    Mul#(input_size, SizeOf#(ElementType), input_bits),
    Mul#(input_lines, input_lines, input_size),
    Bits#(out, output_bits),
    Mul#(TMul#(output_size, 32), output_channels, output_bits),
    Mul#(output_lines, output_lines, output_size),
    // 2D vectors required
    PrimSelectable#(in, Vector::Vector#(input_lines, ElementType)),
    PrimSelectable#(out, Vector::Vector#(output_lines, Vector::Vector#(output_lines, ElementType))),
    Add#(output_lines, kernel_size, TAdd#(input_lines, 1)),
    Mul#(kernel_size, kernel_size, kernel_size_2),
    Mul#(kernel_size_2, SizeOf#(ElementType), kernel_size_2_bits),
    Add#(kernel_size, 0, 3),
    PrimUpdateable#(out, Vector::Vector#(output_lines, Vector::Vector#(output_lines, ElementType)))
  );

  FIFOF#(in) fifo_in <- mkFIFOF;
  FIFOF#(out) fifo_out <- mkFIFOF;

  Wire#(in) data_in <- mkWire;

  rule set_data_in;
    data_in <= fifo_in.first;
  endrule

  Wire#(Vector#(output_lines, Vector#(output_lines, Bit#(kernel_size_2_bits)))) cols <- mkWire;
  // Vector#(output_lines, Vector#(output_lines, Reg#(ElementType))) data_out <- replicateM(replicateM(mkReg(0)));
  // Reg#(Vector#(output_lines, Vector#(output_lines, ElementType))) data_out <- mkReg(unpack('0));

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

  LayerData_ifc#(ElementType, kernel_size_2, output_channels) data <- mkLayerData("cnn", layer_name);
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
    let kernel <- data.getWeights();
    let top = fifo_in.first;
    out t = tmp;
    let bias <- data.getBias();
    for (Integer i = 0; i < valueOf(output_lines); i = i + 1) begin
      for (Integer j = 0; j < valueOf(output_lines); j = j + 1) begin
        ElementType s = 0;
        Vector#(kernel_size_2, ElementType) col = unpack(cols[i][j]);
        // for (Integer k = 0; k < valueOf(kernel_size_2); k = k + 1) begin
        //   s = s + ((col[k] * kernel[k])>> q_bits());
        // end
        // s = ((col[0] * kernel[0])>> q_bits());
        s = col[0] * kernel[0];
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