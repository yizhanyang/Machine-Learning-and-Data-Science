function maxi=max_sum(hidd,noisy_img,x,y,para)
   % pixel probability
   maxi=0; max=0;
   for pixel=0:0.03:1
     phi=mvnpdf(pixel-noisy_img(x,y),0,para(1));
     varphi=0;
     for i=-1:1:1
         for j=-1:1:1
             if x+i>0 && x+i<=size(hidd,1) && y+j>0 && y+j<=size(hidd,2) && i*j==0
                 varphi=varphi+mvnpdf(hidd(x+i,y+j)-pixel,0,para(2));
             end
         end
     end
     sum_pro=phi+varphi;
     if(sum_pro>max)
         max=sum_pro;
         maxi=pixel;
     end
   end
end