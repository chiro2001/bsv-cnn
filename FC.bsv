import Vector::*;
import FIFOF::*;
import FixedPoint::*;

import Utils::*;
import Data::*;
import Layers::*;
import Config::*;

typedef 64 HiddenSize;

module mkTb();

Layer#(Vector#(TMul#(InputWidth, InputHeight), ElementType), Vector#(HiddenSize, ElementType)) fc1 <- mkFCLayer("fc", "fc1");
Layer#(Vector#(HiddenSize, ElementType), Vector#(HiddenSize, ElementType)) relu1 <- mkReluLayer;
Layer#(Vector#(HiddenSize, ElementType), Vector#(10, ElementType)) fc2 <- mkFCLayer("fc", "fc2");
Layer#(Vector#(10, ElementType), ResultType) softmax <- mkSoftmaxLayer;

TestData_ifc#(ElementType, InputWidth, InputHeight) input_data <- mkTestData;
FIFOF#(ResultType) targets <- mkSizedFIFOF(5);

Reg#(int) cnt <- mkReg(0);
Integer max_cnt = 800000;
// Integer max_cnt = 3000;

int maxTotal = fromInteger(valueOf(TEST_DATA_SZ));

Reg#(int) totalPut <- mkReg(0);
Reg#(int) total <- mkReg(0);
Reg#(int) correct <- mkReg(0);

rule hello (cnt == 0);
  $display("Hello FC");
endrule

rule stop if (cnt >= fromInteger(max_cnt) || total > maxTotal);
  $display("Stopping, total: %d, correct: %d, accuracy: %d %%", total, correct, correct * 100 / total);
  $finish(0);
endrule

rule inc_cnt (cnt < fromInteger(max_cnt));
  cnt <= cnt + 1;
endrule

rule put_data if (totalPut + 1 < maxTotal);
  let d <- input_data.get;
  match {.target, .data} = d;
  let target_int = elementToInt(target);
  fc1.put(flatten(data));
  ResultType t = truncate(pack(target_int));
  targets.enq(t);
  totalPut <= totalPut + 1;
endrule

rule put_data_relu1;
  let out <- fc1.get;
  relu1.put(out);
endrule

rule put_data_fc2;
  let out <- relu1.get;
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
