[H,f,A,b] = quadpro();
[x,fval,exitflag,output,lambda] = ...
   quadprog(H,f,A,b);

function [H,f,A,b] = quadpro()
H = [1, -1;-1, 2];
f = [-2;-6];
A = [1, 1;-1, 2;2, 1];
b = [2;2;3];
end