import Vector::*;
import FIFO::*;
import Utils::*;
import Data::*;
import Layers::*;

module mkTb();

Layer#(Vector#(784, Int#(8)), Vector#(32, Int#(8))) fc1 <- mkFCLayer("fc1");
Layer#(Vector#(32, Int#(8)), Vector#(10, Int#(8))) fc2 <- mkFCLayer("fc2");
Layer#(Vector#(10, Int#(8)), Bit#(4)) softmax <- mkSoftmaxLayer;

Reg#(int) cnt <- mkReg(0);
Integer max_cnt = 10000;

rule hello (cnt == 0);
  $display("Hello CNN");
endrule

rule stop (cnt >= fromInteger(max_cnt));
  $display("Stopping");
  $finish;
endrule

rule inc_cnt (cnt < fromInteger(max_cnt));
  cnt <= cnt + 1;
endrule

rule put_data;
  $display("[cnt=%x] Putting data", cnt);
  fc1.put(unpack('0));
  // fc1.put(unpack('h2345678765));
endrule

rule put_data_fc2;
  let out <- fc1.get;
  // $display("[cnt=%x] fc1 -> fc2, out=%x", cnt, pack(out));
  fc2.put(out);
endrule

rule put_data_softmax;
  let out <- fc2.get;
  $display("[cnt=%x] fc2 -> softmax, out=%x", cnt, pack(out));
  softmax.put(out);
  // $write("[cnt=%x] Got fc2 data:", cnt);
  // for (Integer i = 0; i < 10; i = i + 1) begin
  //   $write("[%d: %d], ", i, out[i]);
  // end
  // $display("");
endrule

// rule get_data_fc2;
//   Vector#(10, Int#(8)) data <- fc2.get;
//   $write("[cnt=%x] Got fc2 data:", cnt);
//   for (Integer i = 0; i < 10; i = i + 1) begin
//     $write("[%d: %d], ", i, data[i]);
//   end
//   $display("");
//   // $finish;
// endrule

rule get_data_softmax;
  Bit#(4) data <- softmax.get;
  $display("[cnt=%x] Got softmax data: %d", cnt, data);
  // $finish;
endrule

endmodule
