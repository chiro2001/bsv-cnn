import Vector::*;
import Utils::*;
import Data::*;
import Layers::*;

module mkTb();

Layer#(Vector#(784, Int#(8)), Vector#(32, Int#(8))) fc1 <- mkFCLayer("fc1");

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
  $display("Putting data");
  fc1.put(unpack('0));
endrule

rule get_data;
  $display("Getting data");
  Vector#(32, Int#(8)) data <- fc1.get;
  $write("Got data:");
  for (Integer i = 0; i < 32; i = i + 1) begin
    $write(" %d", data[i]);
  end
  $finish;
endrule

endmodule
