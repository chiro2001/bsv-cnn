import Vector::*;
import FIFO::*;
import FIFOF::*;
import BuildVector::*;

import Utils::*;
import Data::*;
import Layers::*;
import Config::*;
import cnn::*;

typedef 3 KernelSize;

module mkTb();

// // (1, 28, 28) -> (8, 26, 26)
// Layer#(
//     Vector#(1, Vector#(InputWidth, Vector#(InputHeight, ElementType))), 
//     Vector#(8, Vector#(TSub#(InputWidth, TSub#(KernelSize, 1)), Vector#(TSub#(InputHeight, TSub#(KernelSize, 1)), ElementType)))
//   ) conv1 <- mkConvLayer("conv1");
// // relu1
// // (8, 26, 26) -> (4, 24, 24)
// // conv2
// // relu2
// // (4, 24, 24) -> (4, 12, 12)
// // max_pooling1
// // (4, 12, 12) -> (576, )
// // flatten
// // (576, ) -> (32, )
// // fc1
// // relu3
// // (32, ) -> (10, )
// // fc2

// (1, 28, 28) -> (4, 26, 26)
Layer#(
    Vector#(1, Vector#(28, Vector#(28, ElementType))), 
    Vector#(4, Vector#(26, Vector#(26, ElementType)))
  ) conv1 <- mkConvLayer("conv1");
// (4, 26, 26) -> (4, 13, 13)
Layer#(
  Vector#(4, Vector#(26, Vector#(26, ElementType))),
  Vector#(4, Vector#(13, Vector#(13, ElementType)))
) max_pooling1 <- mkMaxPoolingLayer;
// flatten -> relu -> (676, )
Layer#(Vector#(676, ElementType), Vector#(676, ElementType)) relu1 <- mkReluLayer;
Layer#(Vector#(676, ElementType), Vector#(10, ElementType)) fc1 <- mkFCLayer("fc1");
Layer#(Vector#(10, ElementType), ResultType) softmax <- mkSoftmaxLayer;

TestData_ifc#(ElementType, InputWidth, InputHeight) input_data <- mkTestData;
FIFOF#(ResultType) targets <- mkSizedFIFOF(5);

Reg#(int) cnt <- mkReg(0);
Integer max_cnt = 1000000;
// Integer max_cnt = 3000;

int maxTotal = fromInteger(valueOf(TEST_DATA_SZ));

Reg#(int) totalPut <- mkReg(0);
Reg#(int) total <- mkReg(0);
Reg#(int) correct <- mkReg(0);

rule hello (cnt == 0);
  $display("Hello CNN");
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
  conv1.put(vec(data));
  ResultType t = truncate(pack(target_int));
  targets.enq(t);
  totalPut <= totalPut + 1;
endrule

rule put_data_max_pooling1;
  let out <- conv1.get;
  max_pooling1.put(out);
endrule

rule put_data_relu1;
  let out <- max_pooling1.get;
  relu1.put(flatten3(out));
endrule

rule put_data_fc1;
  let out <- relu1.get;
  fc1.put(out);
endrule

rule put_data_softmax;
  let out <- fc1.get;
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
