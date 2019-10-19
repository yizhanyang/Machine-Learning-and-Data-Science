function [nme] = compute_nme(t,y)
mat = (t-y);
t1=mean(t(:));
y1=mean(y(:));
m = mat/(t1*y1);
nme = mean(m(:));
end

