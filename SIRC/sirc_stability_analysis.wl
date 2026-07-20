(* ::Package:: *)

ClearAll["Global`*"]

(* define the SIRC ODE system *)
dSdt = \[Mu]*(1 - S) - \[Beta]*S*II + \[Gamma]*CC;
dIdt = \[Beta]*S*II + \[Sigma]*\[Beta]*CC*II - (\[Mu] + \[Alpha])*II;
dRdt = (1 - \[Sigma])*\[Beta]*CC*II + \[Alpha]*II - (\[Mu] + \[Delta])*RR;
dCdt = \[Delta]*RR - \[Beta]*CC*II - (\[Mu] + \[Gamma])*CC;

(* POSITIVITY AND UNIQUENESS*)
(* reduce *)
(* set all derivatives to zero *)
eqEqs = {dSdt == 0, dIdt == 0, dRdt == 0, dCdt == 0};

(* divide the I-equation by II (we are assuming I =/ 0) and replace *)
eqI = \[Beta]*S + \[Sigma]*\[Beta]*CC == \[Mu] + \[Alpha];
eqSystem = {dSdt == 0, eqI, dRdt == 0, dCdt == 0};

(* eliminate S, RR, CC *)
reduced = Eliminate[eqSystem, {S, RR, CC}] // Simplify;
Print["\nReduced equation in I only:"]
Print[reduced]

(* reduced is: mu * (stuff) == 0, so let's extract the left side *)
poly = reduced[[1]];
poly = poly/\[Mu] // Expand; (* since mu>0*)

(* collect by powers of I *)
quadratic = Collect[poly, II];
Print["\nQuadratic:"]
Print[quadratic]

(* Coefficients *)
aCoeff = Coefficient[quadratic, II, 2] // Simplify;
bCoeff = Coefficient[quadratic, II, 1] // Simplify;
cCoeff = Coefficient[quadratic, II, 0] // Simplify;

(* a is positive (sum of positive parameters)*)
Print["\na = ", aCoeff] 
Print["b = ", bCoeff]
Print["c = ", cCoeff]





(* Findinstance *)

FindInstance[(quadratic /. II -> 0) (quadratic /. II -> 1) > 0 && 0 < \[Sigma] < 1 && \[Delta] > 0 && \[Gamma] > 0 && \[Alpha] > 0 && \[Beta] > 0 && \[Mu] > 0 && \[Beta]/(\[Alpha] + \[Mu]) > 1, {\[Delta], \[Gamma], \[Sigma], \[Alpha], \[Beta], \[Mu]}]

(*aka.  no parameter set with R0>1 where P(0) and P(1) have the same sign?*)



(* check: when is P(0)< 0 given the parameter restrictions and R_0>1? *)
(* aka. when is the polynomial negative when I=0?*)

Reduce[(quadratic /. II -> 0) < 0 && 0 < \[Sigma] < 1 && \[Delta] > 0 && \[Gamma] > 0 && \[Alpha] > 0 && \[Beta] > 0 && \[Mu] > 0 && \[Beta]/(\[Alpha] + \[Mu]) > 1, {\[Delta], \[Gamma], \[Sigma], \[Alpha], \[Beta], \[Mu]}]

(*output is exactly R_0>1*)


(* check: when is P(1)> 0 given the parameter restrictions and R_0>1? *)
(* aka. when is the polynomial positive when I=1?*)

Reduce[(quadratic /. II -> 1) > 0 && 0 < \[Sigma] < 1 && \[Delta] > 0 && \[Gamma] > 0 && \[Alpha] > 0 && \[Beta] > 0 && \[Mu] > 0 && \[Beta]/(\[Alpha] + \[Mu]) > 1, {\[Delta], \[Gamma], \[Sigma], \[Alpha], \[Beta], \[Mu]}]

(*output is exactly R_0>1*)


(* STABILITY*)
F = {dSdt, dIdt, dRdt, dCdt};
vars = {S, II, RR, CC};
Jac = D[F, {vars}];
Jacmat = Jac//MatrixForm

(* rule from setting derivatives to zero with I != 0 *)
eqRules = {
  \[Beta]*S + \[Sigma]*\[Beta]*CC -> \[Mu] + \[Alpha]
};

(* simplify the jacobian using the equilibrium condition *)
JacEq = Jac //. eqRules // Simplify;
Print["\nJacobian at endemic equilibrium:"]
Print[JacEq // MatrixForm]

(* characteristic polynomial *)
cpoly = CharacteristicPolynomial[JacEq, \[Lambda]] // Expand;
cpoly = Collect[cpoly, \[Lambda]] // Simplify;

(* extract coefficients *)
a1 = Coefficient[cpoly, \[Lambda], 3] // Simplify;
a2 = Coefficient[cpoly, \[Lambda], 2] // Simplify;
a3 = Coefficient[cpoly, \[Lambda], 1] // Simplify;
a4 = Coefficient[cpoly, \[Lambda], 0] // Simplify;

Print["\nCharacteristic polynomial coefficients:"]
Print["a1 = ", a1]
Print["a2 = ", a2]
Print["a3 = ", a3]
Print["a4 = ", a4]

Print["\nCharacteristic polynomial:"]
Print[cpoly]


