import Utils::*;
import Vector::*;
import BRAM::*;

interface LayerData_ifc#(type td, type lines, type depth);
  method ActionValue#(Bit#(TMul#(lines, SizeOf#(td)))) getWeights();
  method ActionValue#(td) getBias();
  method Action weightsStart();
  method Action biasStart();
  method Bit#(TAdd#(TLog#(depth), 1)) getWeightsIndex();
  method Bit#(TAdd#(TLog#(lines), 1)) getBiasIndex();
  method Bool weightsDone();
  method Bool biasDone();
endinterface

module mkLayerData#(parameter String model_name, parameter String layer_name)(LayerData_ifc#(td, lines, depth))
    provisos (
      Bits#(td, sz), 
      Literal#(td), 
      Log#(depth, depth_log),
      Log#(lines, lines_log),
      Mul#(lines, sz, lines_bits)
    );
  Reg#(Bit#(TAdd#(depth_log, 1))) index <- mkReg(fromInteger(valueOf(depth) + 1));
  BRAM1Port#(Bit#(TAdd#(depth_log, 1)), Bit#(lines_bits)) weights;
  Reg#(Bit#(TAdd#(lines_log, 1))) index_bias <- mkReg(fromInteger(valueOf(lines) + 1));

  let weightsDoneBool = index >= fromInteger(valueOf(depth) + 1);
  let weightsWillDoneBool = index >= fromInteger(valueOf(depth));
  let biasDoneBool = index_bias >= fromInteger(valueOf(lines) + 1);
  let biasWillDoneBool = index_bias >= fromInteger(valueOf(lines));

  String weights_path = "data/" + model_name + "-" + layer_name + ".weight.hex";
  weights <- mkBRAM1Server(BRAM_Configure{
    memorySize: valueOf(depth), 
    latency: 1, 
    outFIFODepth: 3, 
    allowWriteResponseBypass:False, 
    loadFormat: tagged Hex weights_path
  });

  String bias_path = "data/" + model_name + "-" + layer_name + ".bias.hex";
  BRAM1Port#(Bit#(TAdd#(lines_log, 1)), td) bias <- mkBRAM1Server(BRAM_Configure{
      memorySize: valueOf(lines), 
      latency: 1, 
      outFIFODepth: 3, 
      allowWriteResponseBypass:False, 
      loadFormat: tagged Hex bias_path
    });

  rule read_weights (!weightsWillDoneBool);
    weights.portA.request.put(BRAMRequest{
      write: False, 
      responseOnWrite: False, 
      address: index, 
      datain: 0
    });
  endrule

  rule read_bias (!biasWillDoneBool);
    bias.portA.request.put(BRAMRequest{
      write: False, 
      responseOnWrite: False, 
      address: index_bias, 
      datain: 0
    });
  endrule

  rule inc_index (!weightsDoneBool);
    index <= index + 1;
  endrule

  rule inc_index_bias (!biasDoneBool);
    index_bias <= index_bias + 1;
  endrule

  method getWeights = weights.portA.response.get;

  method getBias = bias.portA.response.get;

  method Action weightsStart();
    index <= 0;
  endmethod

  method Action biasStart();
    index_bias <= 0;
  endmethod

  method getWeightsIndex = index;
  method getBiasIndex = index_bias;

  method Bool weightsDone() = weightsDoneBool;
  method Bool biasDone() = biasDoneBool;

endmodule