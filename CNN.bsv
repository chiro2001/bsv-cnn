import Vector::*;
import Utils::*;
import Data::*;
import Layers::*;

module mkTb();

Layer#(Vector#(784, Int#(8)), Vector#(32, Int#(8))) fc1 <- mkFCLayer("fc1");

rule hello;
  $display("Hello World!");
  $finish;
endrule

endmodule
