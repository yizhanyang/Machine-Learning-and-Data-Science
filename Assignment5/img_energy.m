function total_energy=img_energy(obser,hidd,para)
  total_energy=0;
  for i=1:size(obser,1)
      for j=1:size(obser,2)
          total_energy=total_energy+pixel_energy(obser, hidd,i,j,para);
      end
  end
end