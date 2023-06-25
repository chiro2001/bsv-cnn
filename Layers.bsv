import FIFOF::*;
import FIFO::*;
import Vector::*;
import FixedPoint::*;
import BuildVector::*;

import Data::*;
import Utils::*;
import Config::*;
import cnn::*;

interface Layer#(type in, type out);
  method Action put(in x);
  method ActionValue#(out) get;
endinterface

interface LayerDirect#(type in, type out);
  method Action put(in x);
  method out get;
endinterface

module mkFCLayer#(parameter String model_name, parameter String layer_name)(Layer#(in, out))
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
  LayerData_ifc#(ElementType, lines, depth) data <- mkLayerData(model_name, layer_name);
  Reg#(Bool) done <- mkReg(True);
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
    let weight <- data.getWeights();
    let top = fifo_in.first;
    Vector#(lines, ElementTmpType) t = tmp;
    let bias <- data.getBias();
    ElementTmpType bias_tmp = elementExtend(bias);
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      ElementTmpType mul = elementMult(top[index], weight[i]);
      t[i] = (index == 0 ? 0 : tmp[i]) + mul + (fromInteger(i) == index ? bias_tmp : 0);
    end
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
    for (Integer i = 1; i < valueOf(input_size); i = i + 1) begin
      if (x[i] > x[y]) begin
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

typedef union tagged {
  struct {
    tc c;
    ti i;
    tj j;
    ElementType x;
    ElementType y;
  } MulReq;
  struct {
    tc c;
    ti i;
    tj j;
    ElementType b;
  } AddReq;
  void FinishReq;
} ConvReq#(type tc, type ti, type tj)
  deriving (Bits, Eq, FShow);

// input shape: (input_channels, input_lines, input_lines)
// output shape: (output_channels, output_lines, output_lines)
module mkConvLayer#(parameter String layer_name)(Layer#(in, out))
// now assuming that stride == 1
  provisos (
    Bits#(in, input_bits),
    Bits#(out, output_bits),
    Mul#(TMul#(output_size, SizeOf#(ElementType)), output_channels, output_bits),
    Mul#(TMul#(input_size, SizeOf#(ElementType)), input_channels, input_bits),
    Mul#(input_lines, input_lines, input_size),
    Mul#(output_lines, output_lines, output_size),
    Add#(output_lines, 2, output_lines_2),
    // 3D vectors required
    PrimSelectable#(in, Vector::Vector#(input_lines, Vector::Vector#(input_lines, ElementType))),
    PrimSelectable#(out, Vector::Vector#(output_lines, Vector::Vector#(output_lines, ElementType))),
    Add#(output_lines, kernel_size, TAdd#(input_lines, 1)),
    Mul#(kernel_size, kernel_size, kernel_size_2),
    // Mul#(kernel_size_2, SizeOf#(ElementType), kernel_size_2_bits),
    // Add#(kernel_size, 0, 3),
    PrimUpdateable#(out, Vector::Vector#(output_lines, Vector::Vector#(output_lines, ElementType)))
  );

  FIFOF#(in) fifo_in <- mkFIFOF;
  FIFOF#(out) fifo_out <- mkFIFOF;

  Reg#(Bit#(TAdd#(1, TLog#(TMul#(output_size, output_channels))))) col_i <- mkReg(unpack('0));

  FIFO#(Vector#(TMul#(kernel_size_2, input_channels), ElementType)) cols <- mkFIFO;

  rule img2col;
    $display("[img2col] col_i = %d", col_i);
    let data_in = fifo_in.first;
    Vector#(TMul#(kernel_size_2, input_channels), ElementType) col = unpack('0);
    let x = col_i / fromInteger(valueOf(output_lines));
    let y = col_i % fromInteger(valueOf(output_lines));
    for (Integer channel = 0; channel < valueOf(input_channels); channel = channel + 1) begin
      for (Integer i = 0; i < valueOf(kernel_size); i = i + 1) begin
        for (Integer j = 0; j < valueOf(kernel_size); j = j + 1) begin
          col[channel * valueOf(kernel_size_2) + i * valueOf(kernel_size) + j] = data_in[channel][fromInteger(i) + x][fromInteger(j) + y];
        end
      end
    end
    cols.enq(col);
    if (col_i < fromInteger(valueOf(TMul#(output_size, output_channels)))) begin
      col_i <= col_i + 1;
    end
    else begin
      col_i <= 0;
    end
  endrule

  // total: output_size
  FIFO#(Vector#(TMul#(kernel_size_2, output_channels), ElementType)) col_mul_results <- mkFIFO;

  let weights_data = cnn_conv1_weight();
  let bias_data = cnn_conv1_bias();

  rule calculate_mul_col;
    $display("calculate_mul_col");
    let top = cols.first;
    cols.deq;
    Vector#(TMul#(kernel_size_2, output_channels), ElementType) col = unpack('0);
    for (Integer channel = 0; channel < valueOf(output_channels); channel = channel + 1) begin
      let weights = weights_data[channel];
      for (Integer i = 0; i < valueOf(kernel_size_2); i = i + 1) begin
        col[channel * valueOf(kernel_size_2) + i] = elementTruncate(elementMult(top[i], weights[i]));
      end
    end
    col_mul_results.enq(col);
  endrule

  // total: output_size
  FIFO#(Vector#(output_channels, ElementType)) col_acc_results <- mkFIFO;

  rule calculate_acc_col;
    $display("calculate_acc_col");
    let top = col_mul_results.first;
    col_mul_results.deq;
    Vector#(output_channels, ElementType) sum = unpack('0);
    for (Integer channel = 0; channel < valueOf(output_channels); channel = channel + 1) begin
      for (Integer i = 0; i < valueOf(kernel_size_2); i = i + 1) begin
        sum[channel] = sum[channel] + top[channel * valueOf(kernel_size_2) + i];
      end
      sum[channel] = sum[channel] + bias_data[channel];
    end
    col_acc_results.enq(sum);
  endrule

  Reg#(out) tmp <- mkReg(unpack('0));

  Reg#(Bit#(TLog#(output_size))) output_cnt <- mkReg(0);

  rule cols_to_output;
    let top = col_acc_results.first;
    col_acc_results.deq;
    $display("[cols_to_output] output_cnt = %d", output_cnt);
    let x = output_cnt / fromInteger(valueOf(output_lines));
    let y = output_cnt % fromInteger(valueOf(output_lines));
    out upd = tmp;
    if (output_cnt == 0) begin
      upd = unpack('0);
    end
    for (Integer channel = 0; channel < valueOf(output_channels); channel = channel + 1) begin
      upd[channel][x][y] = top[channel];
    end
    if (output_cnt == fromInteger(valueOf(output_size) - 1)) begin
      output_cnt <= 0;
      tmp <= unpack('0);
      fifo_out.enq(upd);
      fifo_in.deq;
    end
    else begin
      output_cnt <= output_cnt + 1;
      tmp <= upd;
    end
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

module mkMaxPoolingLayer(Layer#(in, out))
  provisos (
    Bits#(in, input_bits),
    Bits#(out, output_bits),
    Mul#(TMul#(output_size, SizeOf#(ElementType)), output_channels, output_bits),
    Mul#(TMul#(input_size, SizeOf#(ElementType)), input_channels, input_bits),
    Mul#(input_lines, input_lines, input_size),
    Mul#(output_lines, output_lines, output_size),
    // decrese size by scale (2)
    Mul#(output_lines, scale, input_lines),
    // same channels
    Add#(input_channels, 0, output_channels),
    // 3D vectors required
    PrimSelectable#(in, Vector::Vector#(input_lines, Vector::Vector#(input_lines, ElementType))),
    PrimSelectable#(out, Vector::Vector#(output_lines, Vector::Vector#(output_lines, ElementType))),
    PrimUpdateable#(out, Vector::Vector#(output_lines, Vector::Vector#(output_lines, ElementType)))
  );

  FIFOF#(in) fifo_in <- mkFIFOF;
  FIFOF#(out) fifo_out <- mkFIFOF;

  rule handle;
    out o = unpack('0);
    let top = fifo_in.first;
    fifo_in.deq;

    for (Integer channel = 0; channel < valueOf(input_channels); channel = channel + 1) begin
      for (Integer i = 0; i < valueOf(output_lines); i = i + 1) begin
        for (Integer j = 0; j < valueOf(output_lines); j = j + 1) begin
          let x = i * valueOf(scale);
          let y = j * valueOf(scale);
          ElementType max = top[channel][x][y];
          for (Integer k = 0; k < valueOf(scale); k = k + 1) begin
            for (Integer l = 0; l < valueOf(scale); l = l + 1) begin
              if (top[channel][x + k][y + l] > max) begin
                max = top[channel][x + k][y + l];
              end
            end
          end
          o[channel][i][j] = max;
        end
      end
    end

    fifo_out.enq(o);
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
