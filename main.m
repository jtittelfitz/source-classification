%for exact_speed = 1:1
%test_set = [1, exact_speed];
%for approx_speed = 1

%%%%%%%%%%%%%%%%%%%%   
%%  Initialization
%%%%%%%%%%%%%%%%%%%%

% key variables
num_sources = 3;
randomize_sources = true;
exact_speed = 1; approx_speed = 1; % speeds indexed in speeds.m
disp_eps = 0.03;            % zero tolerance for displacement (bounds displacement close to zero)
vel_eps  = 0.5;             % zero tolerance for velocity (bounds velocity away from zero)
                            % these parameters need some tuning, seems to
                            % depend on resolution. 0.02 - 0.07 and 0.4-0.5
                            % seem to work well for resolution = 50
                            %
interpolate = false;        % reinterpolate boundary data for new time grid
full_output = false;        % additional output features: identification of sources, movie of wave after source subtraction
debug = false;              % debugging options
write_results = true;       % save solution and error
naive_ident = false;        % identify sources using max mask size
machine_learning = true;
if machine_learning
    train = true;
    learn_on_mask = true;
    training_examples = 20;
    test_examples = 5;
    mld = cell(2,1);
    mld{1} = zeros(1);      % training data
    mld{2} = zeros(1);      % test data
end

for run = 1:training_examples + test_examples
    
if run == training_examples + 1; train = false; end

clear source; clear sensor; clear kgrid; clear sensor_data;

% initialize space grid
resolution = 50; width = 2.1;          % grid represents [-width,width]^2 with <resolution> # of points per unit
Nx = ceil(2*width*resolution); Ny = Nx; % number of points in (one spatial direction of) grid
dx = 1/resolution; dy = dx;             % spatial grid spacing
kgrid = makeGrid(Nx, dx, Ny, dy);

% initialize representation of solution 
% (sources will be color-coded by time they occur)
solution = -10*zeros(Nx);
solution_t = zeros(num_sources,1);

% initialize sound speed 
medium.sound_speed = speeds(kgrid,exact_speed);
medium.density = ones(Nx);

% initialize time grid
t_end = 4;
[kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed, [], t_end);
Nt = size(kgrid.t_array,2);

% initialize boundary: circle of radius 2
sensor.mask = makeCircle(Nx, Ny, 0, 0, 2/dx);

% initialize sources
if randomize_sources  
    separation = -1;    
    while separation <= 0
        t_j = randi([5,floor(0.25*Nt)-1-5],1,num_sources)*dt;   %#ok<*UNRCH> % source times    
        centers = randi([-ceil(Nx/(2*width)),floor(Nx/(2*width))],num_sources,2)*dx; % source centers
        r = randi([floor(0.2/dx),floor(0.25/dx)],[1,num_sources])*dx;  % source radii
        separation = min(min(stdistances(t_j,centers,r,1))); 
    end
else
    t_j = [0.35, 0.6, 0.85];                      % source times
    centers = [-0.75,-0.75; 0.,0.; 0.75,0.75];  % source centers
    r = [0.20, 0.20, 0.20];                     % source radii
end
t_j_grid = floor(t_j / dt) + 1; % grid points corresponding to source times

%%%%%%%%%%%%%%%%%%%%  
%%  Simulation
%%%%%%%%%%%%%%%%%%%%
fprintf('\r ### Beginning Simulation of Source Problem %d of %d ### \r\r',run,training_examples + test_examples);

% simulate first source
source.p0 = (kgrid.x - centers(1,1)).^2 + (kgrid.y - centers(1,2)).^2 < r(1)^2;
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor,'PlotSim',false); 

% shift first source in time, initialize boundary data Lambda
Lambda = [zeros(size(sensor_data,1), t_j_grid(1)), sensor_data(:,1:Nt - t_j_grid(1))];

for j = 2:num_sources
    % simulate jth source
    source.p0 = (kgrid.x - centers(j,1)).^2 + (kgrid.y - centers(j,2)).^2 < r(j)^2;
    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor,'PlotSim',false); 

    % shift jth source in time, add effect to Lambda
    Lambda = Lambda + [zeros(size(sensor_data,1), t_j_grid(j)), sensor_data(:,1:Nt - t_j_grid(j))];    
end

%%%%%%%%%%%%%%%%%%%%   
%%  Reconstruction
%%%%%%%%%%%%%%%%%%%%
fprintf('\r ### Beginning Reconstruction from Boundary %d ### \r\r', run);
% save this for interpolation
if interpolate; [x,t] = ndgrid(1:size(Lambda,1),1:length(kgrid.t_array)); end

% reinitialize grid
kgrid = makeGrid(Nx, dx, Ny, dy);

% initialize approximate speed
medium.sound_speed = speeds(kgrid,approx_speed);
medium.density = ones(Nx);

% reinitialize time grid
t_end = 4;
[kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed, [], t_end);
Nt = size(kgrid.t_array,2);

% interpolate boundary data to new time grid
if interpolate 
    t_j_grid = floor(t_j / dt) + 1;
    [xq,tq] = ndgrid(1:size(Lambda,1),1:length(kgrid.t_array));
    Lambda = interpn(x,t,Lambda,xq,tq); 
    clear x, clear t, clear xq, clear tq;
end

% initialize boundary data
clear source;
source.p_mask = sensor.mask;
source.p = flip(Lambda,2);
source.p_mode = 'dirichlet';

% record full time-reversed waveform
sensor.mask = (kgrid.x.^2 + kgrid.y.^2 < 4);

% run time-reversal
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor,'PlotSim',false);%'DisplayMask','off'); 

% transfer time-reversed waveform to arrays v, integrate to compute u
temp = zeros(Nx);
u = zeros(Nx,Nx,Nt); % particle pressure
v = zeros(Nx,Nx,Nt); % pressure derivative (what we have calculated)

temp(sensor.mask == 1) = sensor_data(:,1);    
v(:,:,1) = temp; 
for n = 2:Nt
    temp(sensor.mask == 1) = sensor_data(:,n);    
    v(:,:,n) = temp;     
    u(:,:,n) = u(:,:,n-1) + 0.5*dt*(v(:,:,n-1) + v(:,:,n));    
end

v = flip(v,3);
u = flip(u,3);

if full_output
    for n = 1:floor(Nt/4) 
        subplot(1,2,1); imagesc(u(:,:,n)); colorbar;
        subplot(1,2,2); imagesc(v(:,:,n)); colorbar; drawnow   
    end
end

if machine_learning
    ind = 1;
    if ~train; ind = 2; end

     if learn_on_mask
        a = (abs(u(:,:,t_j_grid(1))) < disp_eps).*(abs(v(:,:,t_j_grid(1))) > vel_eps);
     else a = v(:,:,t_j_grid(1)); 
     end      
    % generate positive examples
     a = [1; a(:)]';
     if size(mld{ind},1) == 1
         mld{ind} = a;
     else
         mld{ind} = [mld{ind}; a];
     end
     if learn_on_mask
        a = (abs(u(:,:,t_j_grid(2))) < disp_eps).*(abs(v(:,:,t_j_grid(2))) > vel_eps);
     else a = v(:,:,t_j_grid(2)); 
     end      
     a = [1; a(:)]';
     mld{ind} = [mld{ind}; a];
     if learn_on_mask
        a = (abs(u(:,:,t_j_grid(3))) < disp_eps).*(abs(v(:,:,t_j_grid(3))) > vel_eps);
     else a = v(:,:,t_j_grid(3)); 
     end
     a = [1; a(:)]';
     mld{ind} = [mld{ind}; a];  
    % generate random examples 
     for i = 1:10
         n = randi(Nt);
         if learn_on_mask; b = (abs(u(:,:,n)) < disp_eps).*(abs(v(:,:,n)) > vel_eps);
         else b = v(:,:,n); 
         end
         b = [any(t_j_grid == n); b(:)]';    
         mld{ind} = [mld{ind}; b]; 
     end
end

if naive_ident
    %%%%%%%%%%%%%%%%%%%%   
    %%  Ident. Sources
    %%%%%%%%%%%%%%%%%%%%
    fprintf('\r ### Beginning Identification of Sources ### \r\r');

    % calculate energy for comparison (energy should decrease after each source
    % is subtracted)
    KE = kinetic_energy(v,dx);
    PE = potential_energy(u,dx,medium.sound_speed);
    orig_energy = max(KE + PE);
    energy = zeros(num_sources,1);

    % eventually this could be changed to speed things up,
    % as we only need to search the first part of the time interval for sources.
    %
    % reinitialize grid
    %kgrid = makeGrid(Nx, dx, Ny, dy);

    % initialize approximate speed
    %medium.sound_speed = speeds(kgrid,approx_speed);
    %medium.density = ones(Nx);

    % reinitialize time grid (for shorter simulation time)
    %t_end = 2;
    %[kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed, [], t_end);
    %Nt = size(kgrid.t_array,2);

    for i = 1:num_sources
        clear source;
        mask_size = zeros(Nt,1);
        if i > 1; fprintf('\r ### Finding next source ### \r\r'); end
        % identify time corresponding to largest mask
        for n = 1:Nt
            disp_mask = (abs(u(:,:,n)) < disp_eps);
            vel_mask  = (abs(v(:,:,n)) > vel_eps);
            source_mask = disp_mask.*vel_mask;        
            mask_size(n) = sum(sum(source_mask));        
        end

        [~,ind] = max(KE);
        [~,t_max] = max(mask_size);   
        %t_max = ind;           %% alternate choice for source time
        %t_max = t_j_grid(i);   %% this choice is cheating obviously
        fprintf('\r * Max mask size found at %d / KE max at %d  \r\r',t_max,ind);    
        if debug; disp(mask_size(t_j_grid(1)-5:t_j_grid(1)+5)), disp(mask_size(t_j_grid)); keyboard; end     

        % use this time to set source mask and define source
        source_mask = (abs(u(:,:,t_max)) < disp_eps).*(abs(v(:,:,t_max)) > vel_eps);
        source.p0 = v(:,:,t_max).*source_mask;   
        if full_output; imagesc(v(:,:,t_max).*source_mask); fprintf('press any key to continue... \r'); pause; end    

        % run simulation (source artificially at t = 0)
        sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor);

        % shift to source time, add even reflection in time
        temp = zeros(size(sensor_data));
        temp(:,t_max:Nt) = sensor_data(:,1:Nt - t_max + 1);    
        temp(:,1:t_max-1) = flip(sensor_data(:,1:t_max - 1),2); % even reflection in time w.r.t. t_j
        sensor_data = temp;

        % subtract effect of this source
        temp = zeros(Nx);
        for n = 1:Nt
            temp(sensor.mask == 1) = sensor_data(:,n);
            v(:,:,n) = v(:,:,n) - temp;                         
        end

        % integrate to update u
        w = flip(v,3);    
        u(:,:,1) = zeros(Nx);
        for n = 2:Nt
            u(:,:,n) = u(:,:,n-1) + 0.5*dt*(w(:,:,n-1) + w(:,:,n));          
        end            
        u = flip(u,3);
        clear w;

        if full_output
            cmin = 1.1*min(min(min(u))); cmax = 1.1*max(max(max(u)));
            for n = 1:floor(Nt/4)   
                subplot(1,2,1); imagesc(u(:,:,n)); caxis([cmin cmax]),colorbar;
                subplot(1,2,2); imagesc(v(:,:,n)); colorbar; drawnow   
            end
        end

        % calculate change in total energy
        KE = kinetic_energy(v,dx);
        PE = potential_energy(u,dx,medium.sound_speed);
        energy(i) = max(KE + PE)/orig_energy;
        fprintf('\r Relative energy: %f \r',energy(i));

        % add to "solution". not truly a solution; sources in "solution" will get color-coded by time they occur
        solution_t(i) = (t_max-1)*dt;
        solution = bsxfun(@max,solution,((t_max - 1)*dt)*(source_mask)); solution(solution == 0) = -0.5; 
    end
    solution_t = sort(solution_t);
    fprintf('\r Source found at: %f, actual time: %f \r',[solution_t t_j']');
    figure; imagesc(solution); colorbar

    e = solution_t -t_j';
    e = sum(abs(e))/num_sources;
    fprintf('\r relative, average error (in times) %f \r\r',e);

    if write_results
        filename1 = sprintf('resolution%d/solution_nointerp_c%d_c%d.mat',resolution,exact_speed,approx_speed);
        filename2 = sprintf('resolution%d/energy_error_nointerp_c%d_c%d.mat',resolution,exact_speed,approx_speed);
        filename3 = sprintf('resolution%d/time_error_nointerp_c%d_c%d.mat',resolution,exact_speed,approx_speed);

        save(filename1,'solution');
        save(filename2,'energy');
        save(filename3,'e');
    end
end

end %end of single test

approx_speed_train = mld{1};
approx_speed_test = mld{2};
save('approx_speed_training_data.mat','approx_speed_train','approx_speed_test');

%end
%end
