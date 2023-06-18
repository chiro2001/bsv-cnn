import Utils::*;
import Data::*;

module mkTb();

LayerData_ifc fc1 <- mkLayerData(32, 28 * 28, "fc1");

rule hello;
  $display("Hello World!");
  $finish;
endrule

endmodule
