import BRAM::*;
import Vector::*;



module mkTb();

  Reg#(Bit#(10)) cnt <- mkReg(0);
  rule stop;
    cnt <= cnt + 1;
    $display("cnt = %d", cnt);
    if (cnt == 30) $finish;
  endrule

  String test_bias_data_file = "../data/fc-fc1.bias.hex";
  BRAM1Port#(Bit#(10), Bit#(16)) test_bias_data <- mkBRAM1Server(BRAM_Configure{
    memorySize: 32, 
    latency: 1, 
    outFIFODepth: 3, 
    allowWriteResponseBypass:False, 
    loadFormat: tagged Hex test_bias_data_file
  });

  String test_weight_path = "../data/fc-fc1.weight/";
  Vector#(32, BRAM1Port#(Bit#(10), Bit#(16))) weights;
  Vector#(10, String) digitals;
  digitals[0] = "0";
  digitals[1] = "1";
  digitals[2] = "2";
  digitals[3] = "3";
  digitals[4] = "4";
  digitals[5] = "5";
  digitals[6] = "6";
  digitals[7] = "7";
  digitals[8] = "8";
  digitals[9] = "9";
  for (Integer i = 0; i < 32; i = i + 1) begin
    Integer ii = i;
    String path_num = "";
    for (Integer j = 0; j < 3; j = j + 1) begin
      if (ii > 0) begin
        Integer digit = mod(ii, 10);
        path_num = digitals[digit] + path_num;
        ii = div(ii, 10);
      end
    end
    if (path_num == "") path_num = "0";
    let path = test_weight_path + path_num;
    weights[i] <- mkBRAM1Server(BRAM_Configure{
      memorySize: 784, 
      latency: 1, 
      outFIFODepth: 3, 
      allowWriteResponseBypass:False, 
      loadFormat: tagged Hex path
    });
  end

  rule read;
    test_bias_data.portA.request.put(BRAMRequest{
      write: False, 
      responseOnWrite: False, 
      address: cnt, 
      datain: 0
    });
    for (Integer i = 0; i < 32; i = i + 1) begin
      weights[i].portA.request.put(BRAMRequest{
        write: False, 
        responseOnWrite: False, 
        address: cnt, 
        datain: 0
      });
    end
  endrule

  rule test_read_bias;
    Bit#(16) bdata <- test_bias_data.portA.response.get();
    $display("bias = %d", bdata);
    for (Integer i = 0; i < 32; i = i + 1) begin
      Bit#(16) wdata <- weights[i].portA.response.get();
      Int#(16) d = unpack(wdata);
      $write("%d ", d);
    end
    $display("");
  endrule
endmodule