function y=gaussian_process_kernel(x1,x2,theta)
      temp1=norm(x1-x2,2);
      temp2=x1*x2';
      y=theta(1)*exp(-theta(2)*temp1*temp1/2)+theta(3)+theta(4)*temp2;
end