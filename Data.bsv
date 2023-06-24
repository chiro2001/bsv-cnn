import Vector::*;
import BRAM::*;
import FIFOF::*;
import Utils::*;
import Config::*;

// try to use typeclass to implement Literal#(Vector#(n, td)), but failed

// typedef struct {
//   Vector#(n, td) v;
// } LiteralVector#(numeric type n, type td) 
//   deriving (Bits, Literal);

// typeclass LiteralVector#(numeric type n, type td)
//   provisos (Vector#(n, td), Bits#(td, tdSz), Bits#(LiteralVector#(n, td), TMul#(n, SizeOf#(td))), Literal#(td));
//   function LiteralVector#(n, td) fromInteger(Integer x);
//   function Bool inLiteralRange(LiteralVector#(n, td) target, Integer i);
// endtypeclass


interface LayerData_ifc#(type td, type lines, type depth);
  method ActionValue#(Vector#(lines, td)) getWeights();
  method ActionValue#(td) getBias();
  method Action weightsStart();
  method Action biasStart();
  method Action weightsInc();
  method Action biasInc();
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

  method ActionValue#(Vector#(lines, td)) getWeights();
    Bit#(lines_bits) ret <- weights.portA.response.get;
    return unpack(ret);
  endmethod

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

  method Action weightsInc() if (!weightsDoneBool);
    index <= index + 1;
  endmethod

  method Action biasInc() if (!biasDoneBool);
    index_bias <= index_bias + 1;
  endmethod

endmodule

interface TestData_ifc#(type td, type width);
  method ActionValue#(Tuple2#(td, Vector#(width, Vector#(width, td)))) get;
endinterface

module mkTestData(TestData_ifc#(td, width))
  provisos (
    Bits#(td, sz), 
    Literal#(td),
    Bits#(Vector::Vector#(width, Vector::Vector#(width, td)), data_sz),
    Add#(data_sz, sz, tot_sz)
  );
  let data_path = "data/test_input.data.hex";
  let target_path = "data/test_input.target.hex";
  let size = valueOf(TEST_DATA_SZ);
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
  FIFOF#(Tuple2#(td, Vector::Vector#(width, Vector::Vector#(width, td)))) fifo <- mkFIFOF;
  Reg#(Bit#(10)) index <- mkReg(0);

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
  endrule

  rule insert_data;
    let d <- data.portA.response.get;
    let data = unpack(d);
    let t <- target.portA.response.get;
    let target = unpack(t);
    fifo.enq(tuple2(target, data));
  endrule

  method ActionValue#(Tuple2#(td, Vector#(width, Vector#(width, td)))) get;
    fifo.deq;
    index <= index + 1;
    return fifo.first;
  endmethod
endmodule