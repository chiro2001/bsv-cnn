import Vector::*;
import FIFOF::*;
import Utils::*;
import Data::*;
import Layers::*;

module mkTb();

Layer#(Vector#(784, Int#(32)), Vector#(32, Int#(32))) fc1 <- mkFCLayer("fc1");
// Layer#(Vector#(32, Int#(32)), Vector#(32, Int#(32))) relu1 <- mkReluLayer;
Layer#(Vector#(32, Int#(32)), Vector#(10, Int#(32))) fc2 <- mkFCLayer("fc2");
Layer#(Vector#(10, Int#(32)), Int#(32)) softmax <- mkSoftmaxLayer;

TestData_ifc#(Int#(32), 28) input_data <- mkTestData;
FIFOF#(Int#(32)) targets <- mkSizedFIFOF(3);

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
  // fc1.put(unpack('0));
  // fc1.put(unpack('h2345678765));
  // match {.target, .data} <- input_data.get;
  let d <- input_data.get;
  // Tuple2#(Int#(32), Vector::Vector#(28, Vector::Vector#(28, Int#(32)))) d_pack = unpack(d);
  Tuple2#(Int#(32), Vector::Vector#(784, Int#(32))) d_pack = unpack(d);
  match {.target, .data} = d_pack;
  let real_target = target >> q_bits();
  fc1.put(data);
  targets.enq(real_target);
  // if (targets.notFull) begin
  //   $display("[cnt=%x] Putting target %d", cnt, real_target);
  // end
  // fc1.put(unpack('1));
  // $display("[cnt=%x] Putting target %d", cnt, real_target);
endrule

// rule put_data_relu1;
//   let out <- fc1.get;
//   // $display("[cnt=%x] fc1 -> relu1, out=%x", cnt, pack(out));
//   relu1.put(out);
// endrule

rule put_data_fc2;
  // let out <- relu1.get;
  let out <- fc1.get;
  // $display("[cnt=%x] relu1 -> fc2, out=%x", cnt, pack(out));
  fc2.put(out);
endrule

rule put_data_softmax;
  let out <- fc2.get;
  // $display("[cnt=%x] fc2 -> softmax, out=%x", cnt, pack(out));
  softmax.put(out);
  // $write("[cnt=%x] Got fc2 data:", cnt);
  // for (Integer i = 0; i < 10; i = i + 1) begin
  //   $write("[%d: %d], ", i, out[i]);
  // end
  // $display("");
endrule

// rule get_data_fc2;
//   Vector#(10, Int#(32)) data <- fc2.get;
//   $write("[cnt=%x] Got fc2 data:", cnt);
//   for (Integer i = 0; i < 10; i = i + 1) begin
//     $write("[%d: %d], ", i, data[i]);
//   end
//   $display("");
//   // $finish;
// endrule

rule get_data_softmax;
  // Bit#(4) data <- softmax.get;
  // Int#(4) data_idx = unpack(data);
  // Int#(32) real_data = extend(data_idx);
  Int#(32) real_data <- softmax.get;
  Int#(32) target = targets.first;
  targets.deq;
  // $display("[cnt=%x] Got softmax data: %d", cnt, data);
  // $finish(0);
  $write("[cnt=%x] Got target: %d, pred: %d, ", cnt, target, real_data);
  if (real_data == target) begin
    $display("correct");
    correct <= correct + 1;
  end else begin
    $display("wrong");
  end
  total <= total + 1;
endrule

endmodule
