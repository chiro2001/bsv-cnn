// typedef 20 Q_BITS;
// typedef 32 Q_WORD;

typedef 8 Q_BITS;
typedef 16 Q_WORD;

// typedef 64 TEST_DATA_SZ;
typedef 1000 TEST_DATA_SZ;

`define USE_FIXED_POINT

`ifdef USE_FIXED_POINT

import FixedPoint::*;

typedef FixedPoint#(TSub#(Q_WORD, Q_BITS), Q_BITS) ElementType;
typedef FixedPoint#(TMul#(2, TSub#(Q_WORD, Q_BITS)), TMul#(2, Q_BITS)) ElementTmpType;

function ElementTmpType elementMult(ElementType a, ElementType b);
  return fxptMult(a, b);
endfunction

function ElementTmpType elementExtend(ElementType a);
  return fxptSignExtend(a);
endfunction

function ElementType elementTruncate(ElementTmpType a);
  return fxptTruncateSat(Sat_Bound, a);
endfunction

function Int#(Q_WORD) elementToInt(ElementType a);
  return extend(fxptGetInt(a));
endfunction

`else

typedef Int#(Q_WORD) ElementType;
typedef Int#(TMul#(2, Q_WORD)) ElementTmpType;

function ElementTmpType elementMult(ElementType a, ElementType b);
  return signedMul(a, b) >> valueOf(Q_BITS);
endfunction

function ElementTmpType elementExtend(ElementType a);
  return extend(a);
endfunction

function ElementType elementTruncate(ElementTmpType a);
  return truncate(a);
endfunction

function Int#(Q_WORD) elementToInt(ElementType a);
  return extend(a >> valueOf(Q_BITS));
endfunction

`endif

typedef Bit#(4) ResultType;