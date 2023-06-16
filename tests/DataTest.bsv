import BRAM::*;

module mkTb();

  Reg#(Bit#(10)) cnt <- mkReg(0);
  rule stop;
    cnt <= cnt + 1;
    $display("cnt = %d", cnt);
    if (cnt == 10) $finish;
  endrule

  String test_bias_data_file = "../data/fc-fc1.bias.int8";
  BRAM1Port#(Bit#(10), Bit#(8)) test_bias_data <- mkBRAM1Server(BRAM_Configure{
    memorySize: 784, 
    latency: 1, 
    outFIFODepth:3, 
    allowWriteResponseBypass:False, 
    loadFormat: tagged Hex test_bias_data_file
  });

  rule test_read_bias;
    test_bias_data.portA.request.put(BRAMRequest{
      write: False, 
      responseOnWrite: False, 
      address: cnt, 
      datain: 0
    });
    Bit#(8) rdata <- test_bias_data.portA.response.get();
    $display("bias = %d", rdata);
  endrule
endmodule