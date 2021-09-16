using ForwardDiff


function anomaloushall(kx,ky,T,μ,Ham!,H1,H2,H3)
    ϵ = -1:0.1:1
    kb = 8.6*10^(-5) #eV / K
    β = 1/(kb*T) # ≈ 40 at 300 K 
    ħ = 6.582*10^(-16) # eV*s
    Ham!(kx,ky,H1) ; Hx = similar(H1) ; Hy = similar(H1) ; 
    val , vec = eigen!(H1)
    
    curv = similar(val)
    Dx(kx,ky,Ham!,H2,H3,Hx)
    vx = vec'*Hx*vec

    denominator = [ ifelse( n-m ≈ 0, 0, 1/(n-m)^2 ) for n in val , m in val]
    
    Dy(kx,ky,Ham!,H2,H3,Hy)
    vy = vec'*Hy*vec
    
    @. vy *= denominator
    
    @inbounds for c in eachindex(curv)
        curv[c] = -2*imag(sum(vx[c,m]*vy[m,c] for m in 1:length(val)))
    end
    
    @. curv /= T*ħ    # make curv larger, for curv is small
    
    
    nernst = 0
    @inbounds for n in 1:length(val)
        f = exp(β*(val[n] - μ)) |> x -> ifelse(x≈0, 0, x)
        nernst += (val[n]-μ)/( 1+f )+log( 1+1.0/f)/β |>  x -> x*curv[n]
    end
    -nernst
end    