function Berry(kx,ky,Ham!,H1,H2,H3)
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
    curv
end    