function w= linear_inference(t,x,lamda)
   X=x.'*x;
   D=ones(size(X));
   temp=pinv(lamda.*D+X);   
   w=temp*(x.')*t;
end