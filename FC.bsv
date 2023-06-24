import Vector::*;
import FIFOF::*;
import FixedPoint::*;

import Utils::*;
import Data::*;
import Layers::*;
import Config::*;

module mkTb();

Layer#(Vector#(784, ElementType), Vector#(32, ElementType)) fc1 <- mkFCLayer("fc1");
// Layer#(Vector#(32, ElementType), Vector#(32, ElementType)) relu1 <- mkReluLayer;
Layer#(Vector#(32, ElementType), Vector#(10, ElementType)) fc2 <- mkFCLayer("fc2");
Layer#(Vector#(10, ElementType), ResultType) softmax <- mkSoftmaxLayer;

TestData_ifc#(ElementType, 28) input_data <- mkTestData;
FIFOF#(ResultType) targets <- mkSizedFIFOF(10);

Reg#(int) cnt <- mkReg(0);
Integer max_cnt = 300000;

Reg#(int) total <- mkReg(0);
Reg#(int) correct <- mkReg(0);

rule hello (cnt == 0);
  $display("Hello FC");
endrule

rule stop (cnt >= fromInteger(max_cnt));
  $display("Stopping, total: %d, correct: %d, accuracy: %d %%", total, correct, correct * 100 / total);
  $finish(0);
endrule

rule inc_cnt (cnt < fromInteger(max_cnt));
  cnt <= cnt + 1;
endrule

rule put_data;
  let d <- input_data.get;
  Tuple2#(ElementType, Vector::Vector#(784, ElementType)) d_pack = unpack(d);
  match {.target, .data} = d_pack;
  let target_int = fxptGetInt(target);
  fc1.put(data);
  ResultType t = truncate(pack(target_int));
  // $display("[cnt=%x] Put data: %d", cnt, t);
  targets.enq(t);
endrule

rule put_data_fc2;
  let out <- fc1.get;
  fc2.put(out);
endrule

rule put_data_softmax;
  let out <- fc2.get;
  softmax.put(out);
endrule

rule get_data_softmax;
  let data <- softmax.get;
  let target = targets.first;
  targets.deq;
  $write("[cnt=%x] Got target: %d, pred: %d, ", cnt, target, data);
  if (data == target) begin
    $display("correct");
    correct <= correct + 1;
  end else begin
    $display("wrong");
  end
  total <= total + 1;
endrule

endmodule
