import FIFOF::*;
import FIFO::*;
import Vector::*;
import FixedPoint::*;

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

module mkConvLayer#(parameter String layer_name)(Layer#(in, out))
// now assuming that stride == 1
  provisos (
    Bits#(in, input_bits),
    Mul#(input_size, SizeOf#(ElementType), input_bits),
    Mul#(input_lines, input_lines, input_size),
    Bits#(out, output_bits),
    Mul#(TMul#(output_size, SizeOf#(ElementType)), output_channels, output_bits),
    Mul#(output_lines, output_lines, output_size),
    Add#(output_lines, 4, output_lines_2),
    // 3D vectors required
    PrimSelectable#(in, Vector::Vector#(input_lines, Vector::Vector#(input_lines, ElementType))),
    PrimSelectable#(out, Vector::Vector#(output_lines, Vector::Vector#(output_lines, ElementType))),
    Add#(output_lines, kernel_size, TAdd#(input_lines, 1)),
    Mul#(kernel_size, kernel_size, kernel_size_2),
    Mul#(kernel_size_2, SizeOf#(ElementType), kernel_size_2_bits),
    // Add#(kernel_size, 0, 3),
    PrimUpdateable#(out, Vector::Vector#(output_lines, Vector::Vector#(output_lines, ElementType)))
  );

  // TODO: deepper i/o fifos
  FIFOF#(in) fifo_in <- mkFIFOF;
  FIFOF#(out) fifo_out <- mkFIFOF;

  // Wire#(in) data_in <- mkWire;

  // rule set_data_in;
  //   data_in <= fifo_in.first;
  // endrule

  // Wire#(Vector#(output_lines, Vector#(output_lines, Bit#(kernel_size_2_bits)))) cols <- mkWire;
  // // Vector#(output_lines, Vector#(output_lines, Reg#(ElementType))) data_out <- replicateM(replicateM(mkReg(0)));
  // // Reg#(Vector#(output_lines, Vector#(output_lines, ElementType))) data_out <- mkReg(unpack('0));

  // rule bind_cols;
  //   Vector#(output_lines, Vector#(output_lines, Bit#(kernel_size_2_bits))) cols_ = unpack('0);
  //   for (Integer i = 0; i < valueOf(output_lines); i = i + 1) begin
  //     for (Integer j = 0; j < valueOf(output_lines); j = j + 1) begin
  //       cols_[i][j] = {
  //         pack(data_in[i][j]),
  //         pack(data_in[i][j + 1]),
  //         pack(data_in[i][j + 2]),
  //         pack(data_in[i + 1][j]),
  //         pack(data_in[i + 1][j + 1]),
  //         pack(data_in[i + 1][j + 2]),
  //         pack(data_in[i + 2][j]),
  //         pack(data_in[i + 2][j + 1]),
  //         pack(data_in[i + 2][j + 2])
  //       };
  //     end
  //   end
  //   cols <= cols_;
  // endrule

  // LayerData_ifc#(ElementType, kernel_size_2, output_channels) data <- mkLayerData("cnn", layer_name);
  // Reg#(Bool) done <- mkReg(True);
  // Reg#(out) tmp <- mkReg(unpack('0));

  // rule start (done && fifo_in.notEmpty && data.weightsDone() && data.biasDone());
  //   // $display("Layer %s start", layer_name);
  //   data.weightsStart();
  //   data.biasStart();
  //   done <= False;
  //   tmp <= unpack('0);
  // endrule

  // rule acc (!done && !data.weightsDone() && !data.biasDone());
  //   let index = data.getWeightsIndex() - 1;
  //   // $display("Layer %s acc weights, index=%x", layer_name, index);
  //   let kernel <- data.getWeights();
  //   let top = fifo_in.first;
  //   out t = tmp;
  //   let bias <- data.getBias();
  //   for (Integer i = 0; i < valueOf(output_lines); i = i + 1) begin
  //     for (Integer j = 0; j < valueOf(output_lines); j = j + 1) begin
  //       ElementType s = 0;
  //       Vector#(kernel_size_2, ElementType) col = unpack(cols[i][j]);
  //       // for (Integer k = 0; k < valueOf(kernel_size_2); k = k + 1) begin
  //       //   s = s + ((col[k] * kernel[k])>> q_bits());
  //       // end
  //       // s = ((col[0] * kernel[0])>> q_bits());
  //       s = col[0] * kernel[0];
  //       t[index][i][j] = s + bias;
  //     end
  //   end
  //   tmp <= t;
  // endrule

  // rule set_done (!done && data.weightsDone() && data.biasDone());
  //   // $display("Layer %s set done, tmp=%x", layer_name, pack(tmp));
  //   done <= True;
  //   fifo_out.enq(tmp);
  //   fifo_in.deq;
  //   tmp <= unpack('0);
  // endrule

  // Reg#(Vector#(output_channels, Vector::Vector#(output_lines, Vector::Vector#(output_lines, ElementTmpType)))) 
  //   tmp <- mkReg(unpack('0));
  Reg#(out) tmp <- mkReg(unpack('0));

  FIFO#(ConvReq#(Bit#(output_channels), Bit#(TLog#(output_lines_2)), Bit#(TLog#(output_lines_2)))) reqPipe <- mkFIFO;

  Reg#(Bit#(output_channels)) c <- mkReg(unpack('0));
  Reg#(Bit#(TLog#(output_lines_2))) i <- mkReg(unpack('0));
  Reg#(Bit#(TLog#(output_lines_2))) j <- mkReg(unpack('0));
  Reg#(Bit#(TLog#(TMul#(output_channels, kernel_size_2)))) xi <- mkReg(unpack('0));

  rule enq_requests;
    let weights = cnn_conv1_weight();
    let top = fifo_in.first;
    Bit#(output_channels) nc = c;
    Bit#(TLog#(output_lines_2)) ni = i;
    Bit#(TLog#(output_lines_2)) nj = j;
    Bit#(TLog#(TMul#(output_channels, kernel_size_2))) nxi = xi + 1;
    Bool done = False;
    if (nxi + 1 == fromInteger(valueOf(TMul#(output_channels, kernel_size_2)))) begin
      nxi = pack('0);
      nj = nj + 1;
      if (nj + 1 == fromInteger(valueOf(output_lines))) begin
        nj = pack('0);
        ni = ni + 1;
        if (ni + 1 == fromInteger(valueOf(output_lines))) begin
          ni = pack('0);
          nc = nc + 1;
          if (nc + 1 == fromInteger(valueOf(output_channels))) begin
            nc = pack('0);
            done = True;
          end
        end
      end
    end
    if (!done) begin
      ElementType xx = top[0][0][0];
      Integer xx_cnt = 0;
      Bool xx_selected = False;
      for (Integer id = 0; id < valueOf(output_lines); id = id + 1) begin
        for (Integer jd = 0; jd < valueOf(output_lines); jd = jd + 1) begin
          if (!xx_selected && fromInteger(xx_cnt) == nxi) begin
            xx = top[nc][ni + fromInteger(id)][nj + fromInteger(jd)];
            xx_selected = True;
          end
        end
      end
      ElementType ww = unpack(pack(weights[nc][nxi]));
      reqPipe.enq(tagged MulReq { c: nc, i: ni, j: nj, x: xx, y: ww });
    end
    else begin
      reqPipe.enq(tagged FinishReq);
    end
  endrule

  rule apply_requests;
    let req = reqPipe.first;
    reqPipe.deq;
    let upd = tmp;
    if (req matches tagged MulReq .mul_req) begin
      let mul_raw = elementMult(mul_req.x, mul_req.y);
      // upd[mul_req.c][mul_req.i][mul_req.j] = tmp[mul_req.c][mul_req.i][mul_req.j] + mul_raw;
      upd[mul_req.c][mul_req.i][mul_req.j] = tmp[mul_req.c][mul_req.i][mul_req.j] + elementTruncate(mul_raw);
    end
    else if (req matches tagged FinishReq) begin
      // out o = unpack('0);
      // for (Integer c = 0; c < valueOf(output_channels); c = c + 1) begin
      //   for (Integer i = 0; i < valueOf(output_lines); i = i + 1) begin
      //     for (Integer j = 0; j < valueOf(output_lines); j = j + 1) begin
      //       o[c][i][j] = elementTruncate(upd[c][i][j]);
      //     end
      //   end
      // end
      let o = tmp;
      fifo_out.enq(o);
      fifo_in.deq;
    end
    tmp <= upd;
  endrule

  let bias_data = cnn_conv1_bias();
  out tmp_init = unpack('0);
  for (Integer channel = 0; channel < valueOf(output_channels); channel = channel + 1) begin
    for (Integer ii = 0; ii < valueOf(output_lines); ii = ii + 1) begin
      for (Integer jj = 0; jj < valueOf(output_lines); jj = jj + 1) begin
        // t[channel][ii][jj] = elementExtend(unpack(pack(bias_data[channel])));
        tmp_init[channel][ii][jj] = unpack(pack(bias_data[channel]));
      end
    end
  end

  method Action put(in x);
    // Vector#(output_channels, Vector::Vector#(output_lines, Vector::Vector#(output_lines, ElementTmpType))) t = unpack('0);
    // out t = unpack('0);
    // for (Integer channel = 0; channel < valueOf(output_channels); channel = channel + 1) begin
    //   for (Integer ii = 0; ii < valueOf(output_lines); ii = ii + 1) begin
    //     for (Integer jj = 0; jj < valueOf(output_lines); jj = jj + 1) begin
    //       // t[channel][ii][jj] = elementExtend(unpack(pack(bias_data[channel])));
    //       t[channel][ii][jj] = unpack(pack(bias_data[channel]));
    //     end
    //   end
    // end
    // tmp <= t;
    tmp <= tmp_init;
    fifo_in.enq(x);
  endmethod

  method ActionValue#(out) get;
    out y = fifo_out.first;
    fifo_out.deq;
    return y;
  endmethod

endmodule