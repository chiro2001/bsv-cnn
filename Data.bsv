import Utils::*;
import Vector::*;
import BRAM::*;

interface LayerData_ifc#(type td, type lines, type depth);
  method Vector#(lines, ActionValue#(td)) getWeight();
  method ActionValue#(td) getBias();
  method Action resetState();
  method Bit#(TLog#(depth)) getIndex();
endinterface

module mkLayerData#(parameter String layer_name)(LayerData_ifc#(td, lines, depth))
    provisos (
      Bits#(td, sz), 
      Literal#(td), 
      Add#(TLog#(lines), v__, depth_log), 
      Log#(depth, depth_log)
    );
  Reg#(Bit#(depth_log)) index <- mkReg(0);
  Wire#(Bit#(depth_log)) addr <- mkDWire(0);
  Vector#(lines, BRAM1Port#(Bit#(depth_log), td)) weights;

  String weights_path = "data/fc-" + layer_name + ".weight/";
  for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
    let path = weights_path + numberToString(i);
    weights[i] <- mkBRAM1Server(BRAM_Configure{
      memorySize: valueOf(depth), 
      latency: 1, 
      outFIFODepth: 3, 
      allowWriteResponseBypass:False, 
      loadFormat: tagged Hex path
    });
  end

  String bias_path = "data/fc-" + layer_name + ".bias.int8";
  BRAM1Port#(Bit#(TLog#(lines)), td) bias <- mkBRAM1Server(BRAM_Configure{
      memorySize: valueOf(lines), 
      latency: 1, 
      outFIFODepth: 3, 
      allowWriteResponseBypass:False, 
      loadFormat: tagged Hex bias_path
    });

  // default value?
  rule set_addr /*(index != 'haaaaaaaa)*/;
    addr <= index;
  endrule

  rule read;
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      weights[i].portA.request.put(BRAMRequest{
        write: False, 
        responseOnWrite: False, 
        address: addr, 
        datain: 0
      });
    end
    // if (addr < fromInteger(valueOf(lines))) begin
    bias.portA.request.put(BRAMRequest{
      write: False, 
      responseOnWrite: False, 
      address: truncate(addr), 
      datain: 0
    });
    // end
  endrule

  rule inc_index;
    index <= (index == (fromInteger(valueOf(depth)) - 1) ? 0 : (index + 1));
  endrule

  function ActionValue#(td) getWeightValue(Integer i);
    return weights[i].portA.response.get();
  endfunction

  method Vector#(lines, ActionValue#(td)) getWeight();
    return map(getWeightValue, genVector);
  endmethod

  method getBias = bias.portA.response.get;

  method Action resetState();
    index <= 0;
  endmethod

  method getIndex = index;

endmodule