function c = speeds(kgrid,c_val)

   if c_val == 1
       c = ones(kgrid.Nx);
   elseif c_val == 2    
       c = 1 + 0.01*sin(kgrid.x) + 0.01*cos(kgrid.y);
   end