function y=gaussian_kernel(x1,x2,sigma)
      temp=norm(x1-x2,2);
      y=exp(-temp*temp/(2*sigma^2));
end