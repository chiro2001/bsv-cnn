// import fc::*;
import Vector::*;

function Vector#(2, Int#(8)) get_test_vector();
  Vector#(2, Int#(8)) vec;
  vec[0] = 1;
  vec[1] = 2;
  return vec;
endfunction

function Vector#(2, Vector#(2, Int#(8))) get_test_vector2();
  Vector#(2, Vector#(2, Int#(8))) vec;
  vec[0][0] = 1;
  vec[0][1] = 2;
  vec[1][0] = 3;
  vec[1][1] = 4;
  return vec;
endfunction

module mkTb();

// function Int#(8)[][] get_test_array();
//   Int#(8) arr[2][2];
//   arr[0][0] = 1;
//   arr[0][1] = 4;
//   arr[1][0] = 5;
//   arr[1][0] = 1;
//   return arr;
// endfunction

// // let bias <- FC_fc1_bias();
// let a <- get_test_array();

let v = get_test_vector();
let v2 = get_test_vector2();

// let weight = fc_fc1_weight();

rule test;
  $display("pass!");
  $finish;
endrule

endmodule