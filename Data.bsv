import Utils::*;
import Vector::*;
import BRAM::*;

interface LayerData_ifc;
  method Vector#(32, ActionValue#(Bit#(8))) getWeight();
  method ActionValue#(Bit#(8)) getBias();
  method Action resetState();
endinterface

module mkLayerData#(parameter Integer lines, parameter Integer depth, parameter String layer_name)(LayerData_ifc);
  Reg#(Bit#(32)) index <- mkReg(0);
  Vector#(32, BRAM1Port#(Bit#(32), Bit#(8))) weights;

  String weights_path = "data/fc-" + layer_name + ".weight/";
  for (Integer i = 0; i < lines; i = i + 1) begin
    let path = weights_path + numberToString(i);
    weights[i] <- mkBRAM1Server(BRAM_Configure{
      memorySize: depth, 
      latency: 1, 
      outFIFODepth: 3, 
      allowWriteResponseBypass:False, 
      loadFormat: tagged Hex path
    });
  end

  String bias_path = "data/fc-" + layer_name + ".bias.int8";
  BRAM1Port#(Bit#(32), Bit#(8)) bias <- mkBRAM1Server(BRAM_Configure{
      memorySize: lines, 
      latency: 1, 
      outFIFODepth: 3, 
      allowWriteResponseBypass:False, 
      loadFormat: tagged Hex bias_path
    });

  rule read;
    for (Integer i = 0; i < lines; i = i + 1) begin
      weights[i].portA.request.put(BRAMRequest{
        write: False, 
        responseOnWrite: False, 
        address: index, 
        datain: 0
      });
    end
    bias.portA.request.put(BRAMRequest{
      write: False, 
      responseOnWrite: False, 
      address: index, 
      datain: 0
    });
    index <= (index == (fromInteger(depth) - 1) ? 0 : (index + 1));
  endrule

  function ActionValue#(Bit#(8)) getWeightValue(Integer i);
    return weights[i].portA.response.get();
  endfunction

  method Vector#(32, ActionValue#(Bit#(8))) getWeight();
    return map(getWeightValue, genVector);
  endmethod

  method getBias = bias.portA.response.get;

  method Action resetState();
    index <= 0;
  endmethod

endmodule