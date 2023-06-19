import Utils::*;
import Vector::*;
import BRAM::*;

interface LayerData_ifc#(type td, type lines, type depth);
  method Vector#(lines, ActionValue#(td)) getWeight();
  method ActionValue#(td) getBias();
  method Action resetState();
  method Bit#(TAdd#(TLog#(depth), 1)) getIndex();
  method Bit#(TAdd#(TLog#(lines), 1)) getIndexLines();
  method Bool weightsDone();
  method Bool biasDone();
  method Bool weightsWillDone();
  method Bool biasWillDone();
endinterface

module mkLayerData#(parameter String layer_name)(LayerData_ifc#(td, lines, depth))
    provisos (
      Bits#(td, sz), 
      Literal#(td), 
      Log#(depth, depth_log),
      Log#(lines, lines_log)
    );
  Reg#(Bit#(TAdd#(depth_log, 1))) index <- mkReg(fromInteger(valueOf(depth)));
  // Wire#(Bit#(TAdd#(depth_log, 1))) addr <- mkDWire(0);
  // Wire#(Bit#(depth_log)) index_next <- mkDWire(0);
  Vector#(lines, BRAM1Port#(Bit#(TAdd#(depth_log, 1)), td)) weights;
  Reg#(Bit#(TAdd#(lines_log, 1))) index_bias <- mkReg(fromInteger(valueOf(lines)));
  // Wire#(Bit#(TAdd#(lines_log, 1))) addr_bias <- mkDWire(0);
  // Wire#(Bit#(lines_log)) index_bias_next <- mkDWire(0);

  let weightsDoneBool = index >= fromInteger(valueOf(depth));
  let biasDoneBool = index_bias >= fromInteger(valueOf(lines));

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
  BRAM1Port#(Bit#(TAdd#(lines_log, 1)), td) bias <- mkBRAM1Server(BRAM_Configure{
      memorySize: valueOf(lines), 
      latency: 1, 
      outFIFODepth: 3, 
      allowWriteResponseBypass:False, 
      loadFormat: tagged Hex bias_path
    });

  // // default value?
  // rule set_addr /*(index != 'haaaaaaaa)*/;
  //   addr <= index;
  //   addr_bias <= index_bias;
  // endrule

  rule read_weights (!weightsDoneBool);
    for (Integer i = 0; i < valueOf(lines); i = i + 1) begin
      weights[i].portA.request.put(BRAMRequest{
        write: False, 
        responseOnWrite: False, 
        address: index, 
        datain: 0
      });
    end
  endrule

  rule read_bias (!biasDoneBool);
    bias.portA.request.put(BRAMRequest{
      write: False, 
      responseOnWrite: False, 
      address: index_bias, 
      datain: 0
    });
  endrule

  // rule inc_index;
  //   let index_max = fromInteger(valueOf(depth) - 1);
  //   let index_next = (addr == index_max ? index_max : (addr + 1));
  //   index <= index_next;
  //   let index_bias_max = fromInteger(valueOf(lines) - 1);
  //   let index_bias_next = (index_bias == index_bias_max ? index_bias_max : (index_bias + 1));
  //   index_bias <= index_bias_next;
  // endrule

  rule inc_index (!weightsDoneBool);
    index <= index + 1;
  endrule

  rule inc_index_bias (!biasDoneBool);
    index_bias <= index_bias + 1;
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
    index_bias <= 0;
  endmethod

  method getIndex = index;
  method getIndexLines = index_bias;

  method Bool weightsDone() = weightsDoneBool;
  method Bool biasDone() = biasDoneBool;
  method Bool weightsWillDone() = index + 1 >= fromInteger(valueOf(depth));
  method Bool biasWillDone() = index_bias + 1 >= fromInteger(valueOf(lines));

endmodule