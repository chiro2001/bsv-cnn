import Vector::*;
import FIFO::*;

import Utils::*;
import Data::*;
import Layers::*;
import Config::*;
import cnn::*;

typedef 3 KernelSize;

module mkTb();

Layer#(
    Vector#(1, Vector#(InputWidth, Vector#(InputHeight, ElementType))), 
    Vector#(8, Vector#(TSub#(InputWidth, TSub#(KernelSize, 1)), Vector#(TSub#(InputHeight, TSub#(KernelSize, 1)), ElementType)))
  ) conv1 <- mkConvLayer("conv1");

Reg#(int) cnt <- mkReg(0);
Integer max_cnt = 100;

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
