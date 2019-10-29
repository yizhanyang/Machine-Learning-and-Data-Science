function e=pixel_energy(obser, hidd,x,y,para)
    h_bias=para(1);beta=para(2); sig1=para(3); sig2=para(4);
    neigh=[-1,0,1;-1,0,1];
    %e=h_bias*hidd(x,y)-eta*obser(x,y)*hidd(x,y);
   % e=(hidd(x,y)-obser(x,y))^2;
    e=log(1+((hidd(x,y)-obser(x,y))^2)/(2*sig1^2));
    for i=1:size(neigh,2)
        for j=1:size(neigh,2)
            if(neigh(1,i)~=0 || neigh(2,j)~=0)
                if(x+neigh(1,i)>=1 && x+neigh(1,i)<=size(hidd,1) && y+neigh(2,j)>=1 && y+neigh(2,j)<=size(hidd,2))
                    %e=e-beta*hidd(x,y)*hidd(x+neigh(1,i),y+neigh(2,j));
                    if(neigh(1,i)*neigh(2,j)==1)
                       %e=e+h_bias*(hidd(x,y)-hidd(x+neigh(1,i),y+neigh(2,j)))^2;
                       e=e+h_bias*log(1+((hidd(x,y)-hidd(x+neigh(1,i),y+neigh(2,j)))^2)/(2*sig2^2));
                    else
                       %e=e+h_bias*beta*(hidd(x,y)-hidd(x+neigh(1,i),y+neigh(2,j)))^2;
                       e=e+h_bias*beta*log(1+((hidd(x,y)-hidd(x+neigh(1,i),y+neigh(2,j)))^2)/(2*sig2^2));
                    end
                end
            end
        end
    end
end