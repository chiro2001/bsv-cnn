import Vector::*;
import BRAM::*;
import FIFOF::*;
import Utils::*;

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
  Reg#(Bit#(TAdd#(lines_log, 1))) index_bias <- mkReg(fromInteger(valueOf(lines) + 1));

  let weightsDoneBool = index >= fromInteger(valueOf(depth) + 1);
  let weightsWillDoneBool = index >= fromInteger(valueOf(depth));
  let biasDoneBool = index_bias >= fromInteger(valueOf(lines) + 1);
  let biasWillDoneBool = index_bias >= fromInteger(valueOf(lines));

  String weights_path = "data/" + model_name + "-" + layer_name + ".weight.hex";
  BRAM1Port#(Bit#(TAdd#(depth_log, 1)), Bit#(lines_bits)) weights <- mkBRAM1Server(BRAM_Configure{
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

interface TestData_ifc#(type td, type width);
  // method ActionValue#(Tuple2#(td, Vector#(width, Vector#(width, td)))) get;
  method ActionValue#(Bit#(SizeOf#(Tuple2#(td, Vector#(width, Vector#(width, td)))))) get;
  // method ActionValue#(Bits#(TAdd#(1, TMul#(width, TMul#(width, SizeOf#(td)))))) get;
endinterface

module mkTestData(TestData_ifc#(td, width))
  provisos (
    Bits#(td, sz), 
    Literal#(td),
    // Literal#(Vector::Vector#(width, Vector::Vector#(width, td))),
    // Literal#(Tuple2#(td, Vector::Vector#(width, Vector::Vector#(width, td)))),
    Bits#(Vector::Vector#(width, Vector::Vector#(width, td)), data_sz),
    Bits#(Tuple2#(td, Vector::Vector#(width, Vector::Vector#(width, td))), tot_sz)
  );
  let data_path = "data/test_input.data.hex";
  let target_path = "data/test_input.target.hex";
  let size = 1000;
  // BRAM1Port#(Bit#(10), Vector#(width, Vector#(width, td))) data
  BRAM1Port#(Bit#(10), Bit#(data_sz)) data
      <- mkBRAM1Server(BRAM_Configure{
    memorySize: size, 
    latency: 1, 
    outFIFODepth: 3, 
    allowWriteResponseBypass:False, 
    loadFormat: tagged Hex data_path
  });
  BRAM1Port#(Bit#(10), Bit#(sz)) target <- mkBRAM1Server(BRAM_Configure{
    memorySize: size, 
    latency: 1, 
    outFIFODepth: 3, 
    allowWriteResponseBypass:False, 
    loadFormat: tagged Hex target_path
  });
  // FIFOF#(Tuple2#(td, Vector#(width, Vector#(width, td)))) fifo <- mkFIFOF;
  FIFOF#(Bit#(tot_sz)) fifo <- mkFIFOF;
  // Reg#(Bit#(10)) index <- mkReg('1);
  Reg#(Bit#(10)) index <- mkReg(0);

  // rule req_data (index != '1);
  rule req_data;
    data.portA.request.put(BRAMRequest{
      write: False, 
      responseOnWrite: False, 
      address: index, 
      datain: 0
    });
    target.portA.request.put(BRAMRequest{
      write: False, 
      responseOnWrite: False, 
      address: index, 
      datain: 0
    });
    // $display("req_data: index=%x", index);
  endrule

  rule insert_data;
    let d <- data.portA.response.get;
    let t <- target.portA.response.get;
    // fifo.enq(tuple2(t, d));
    // fifo.enq(pack(tuple2(t, d)));
    // $display("insert test data, target is %x", t);
    fifo.enq({t, d});
  endrule

  // method ActionValue#(Tuple2#(td, Vector#(width, Vector#(width, td)))) get;
  method ActionValue#(Bit#(tot_sz)) get;
    fifo.deq;
    index <= index + 1;
    return fifo.first;
  endmethod
endmodule