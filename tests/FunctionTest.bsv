function int no_action();
  return 1;
endfunction

module mkTb();

let v = no_action();

rule ok;
  $finish;
endrule
endmodule
