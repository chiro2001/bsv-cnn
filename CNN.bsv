import Vector::*;
import FIFO::*;
import Utils::*;
import Data::*;
import Layers::*;

module mkTb();

Layer#(Vector#(28, Vector#(28, Int#(8))), Vector#(26, Vector#(26, Int#(8)))) conv1 <- mkConvLayer("conv1");

Reg#(int) cnt <- mkReg(0);
Integer max_cnt = 10000;

rule hello (cnt == 0);
  $display("Hello CNN");
endrule

rule stop (cnt >= fromInteger(max_cnt));
  $display("Stopping");
  $finish(0);
endrule

rule inc_cnt (cnt < fromInteger(max_cnt));
  cnt <= cnt + 1;
endrule

rule put_data;
  $display("[cnt=%x] Putting data", cnt);
  conv1.put(unpack('0));
  // fc1.put(unpack('h2345678765));
endrule

// rule get_data_softmax;
//   Bit#(4) data <- softmax.get;
//   $display("[cnt=%x] Got softmax data: %d", cnt, data);
//   // $finish(0);
// endrule

endmodule
