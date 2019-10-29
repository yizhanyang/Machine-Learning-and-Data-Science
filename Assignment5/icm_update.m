function [hidd, total_energy]= icm_update(obser,hidd,x,y,total_energy,para)
   p_energy=pixel_energy(obser,hidd,x,y,para);
   res_energy=total_energy-p_energy;
   new_hidd=hidd;
   if hidd(x,y)<0.2
       poten=[0,0.4];
   elseif hidd(x,y)<0.4
       poten=[0.2,0.6];
   elseif hidd(x,y)<0.6
       poten=[0.4,0.8];
   else
       poten=[0,1];
   end
   for pix_value=poten(1):0.01:poten(2)
     p_energy=pixel_energy(obser,new_hidd,x,y,para);
     new_hidd(x,y)=pix_value;
     new_p_energy=pixel_energy(obser,new_hidd,x,y,para);
     if new_p_energy<p_energy
         total_energy=res_energy+new_p_energy;
         hidd(x,y)=new_hidd(x,y);
     end
   end
end