function x=basis_pursuit(b,A,ep)
% Solve P1 problem via linear programming
%options=optimoptions('linprog','Algorithm','dual-simplex',...
%    'Display','none','OptimalityTolerance',ep);
n=size(A,1); m=size(A,2); I=eye(m);
f=[zeros(1,m), ones(1,m)];
Al=[[I,-I];[-I,-I]];
bl=zeros(2*m,1);
x=linprog(f,Al,bl,[A,zeros(n,m)],b);
x(abs(x)<ep)=0;
x=x(1:m);
end