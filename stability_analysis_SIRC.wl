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

(* STABILITY*)
F = {dSdt, dIdt, dRdt, dCdt};
vars = {S, II, RR, CC};
Jac = D[F, {vars}];

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





\[Lambda] (\[Lambda]+\[Mu]) (\[Gamma]+\[Lambda]+\[Mu]) (\[Delta]+\[Lambda]+\[Mu])+II^2 \[Beta]^2 (\[Lambda]^2+\[Lambda] \[Mu]-\[Alpha] \[Delta] \[Sigma]+(CC \[Beta]+\[Delta]) \[Lambda] \[Sigma]+CC \[Beta] \[Sigma] (\[Mu]+\[Delta] \[Sigma])+S \[Beta] (\[Lambda]+\[Mu]+\[Delta] \[Sigma]))+II \[Beta] (CC \[Beta] \[Gamma] \[Lambda]+\[Gamma] \[Delta] \[Lambda]+\[Gamma] \[Lambda]^2+\[Delta] \[Lambda]^2+2 \[Lambda]^3+CC \[Beta] \[Gamma] \[Mu]+\[Gamma] \[Lambda] \[Mu]+\[Delta] \[Lambda] \[Mu]+4 \[Lambda]^2 \[Mu]+2 \[Lambda] \[Mu]^2+S \[Beta] (\[Gamma]+\[Lambda]+\[Mu]) (\[Delta]+\[Lambda]+\[Mu])+CC \[Beta] \[Gamma] \[Delta] \[Sigma]+CC \[Beta] \[Lambda]^2 \[Sigma]+\[Delta] \[Lambda]^2 \[Sigma]+2 CC \[Beta] \[Lambda] \[Mu] \[Sigma]+\[Delta] \[Lambda] \[Mu] \[Sigma]+CC \[Beta] \[Mu]^2 \[Sigma]+CC \[Beta] \[Delta] \[Lambda] \[Sigma]^2+CC \[Beta] \[Delta] \[Mu] \[Sigma]^2-\[Alpha] \[Delta] (\[Gamma]+(\[Lambda]+\[Mu]) \[Sigma])) /.\[Beta]->R(\[Mu]+\[Alpha])








solutions = Solve[{dSdt == 0, eqI, dRdt == 0, dCdt == 0}, {S, II, RR, CC}];




