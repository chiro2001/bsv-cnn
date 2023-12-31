import Vector::*;

function String numberToString(Integer i);
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
  Integer ii = i;
  String num = "";
  for (Integer j = 0; j < 3; j = j + 1) begin
    if (ii > 0) begin
      Integer digit = mod(ii, 10);
      num = digitals[digit] + num;
      ii = div(ii, 10);
    end
  end
  if (num == "") num = "0";
  return num;
endfunction

// function Integer q_bits();
//   return 20;
// endfunction

function Vector#(m, td) flatten(Vector#(n1, Vector#(n2, td)) x)
    provisos (Mul#(n1, n2, m), Bits#(td, tdSz));
  return unpack(pack(x));
endfunction

function Vector#(m, td) flatten3(Vector#(n3, Vector#(n1, Vector#(n2, td))) x)
    provisos (Mul#(TMul#(n1, n2), n3, m), Bits#(td, tdSz));
  return unpack(pack(x));
endfunction