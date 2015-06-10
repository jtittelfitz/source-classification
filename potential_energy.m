function PE = potential_energy(u,dx,c)
    N = size(u,3);
    PE = zeros(N,1);
    for n = 1:N
        [DX,DY] = gradient(u(:,:,n),dx,dx);
        PE(n)  = (dx^2)*(norm(c.*DX,'fro')^2 + norm(c.*DY,'fro'));
    end    
end