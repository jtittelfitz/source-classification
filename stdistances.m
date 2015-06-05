function sd = stdistances( t_j,centers,r,output )
    if nargin < 4
        output = 1;
    end
    for i = 1:length(t_j)
        for j = 1:length(t_j)
            if i == j
                sd(i,j) = 1000;
            else
                d = sqrt((centers(i,1) - centers(j,1))^2 + (centers(i,2) - centers(j,2))^2);
                sd(i,j) = (d - r(i) - r(j) - abs(t_j(i) - t_j(j)));
            end
        end
    end
    if output
        fprintf('--------------- \r space-time separation: %2.3f \r',min(min(sd)))
    end
end

