function KE = kinetic_energy(v,dx)
    N = size(v,3);
    KE = zeros(N,1);
    for n = 1:N
        KE(n) = dx^2*norm(v(:,:,n),'fro')^2;
    end    
end