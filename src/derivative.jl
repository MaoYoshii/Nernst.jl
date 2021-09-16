using ForwardDiff

export  FermiOccupation,
        ∂ϵFermiOccupation,
        ∂μFermiOccupation,
        ∂TFermiOccupation

function FermiOccupation(ϵ,μ,T) 
    kb = 8.6*10^(-5) #eV / K
    1/(exp( (ϵ-μ)/(kb*T)) +1 )
end

∂ϵFermiOccupation(ϵ,μ,T) = ForwardDiff.derivative( x -> FermiOccupation(x,μ,T))
∂μFermiOccupation(ϵ,μ,T) = ForwardDiff.derivative( x -> FermiOccupation(ϵ,x,T))
∂TFermiOccupation(ϵ,μ,T) = ForwardDiff.derivative( x -> FermiOccupation(ϵ,μ,x))

function Dx(kx,ky,Ham!,Matrix1,Matrix2,diff)
    dϵ = 1<<15
    ϵ = 1/dϵ
    Ham!(kx+ϵ,ky,Matrix1) ; Ham!(kx-ϵ,ky,Matrix2)
    @. diff = (Matrix1-Matrix2)*dϵ/2
end

function Dy(kx,ky,Ham!,Matrix1,Matrix2,diff)
    dϵ = 1<<15
    ϵ = 1/dϵ
    Ham!(kx,ky+ϵ,Matrix1) ; Ham!(kx,ky-ϵ,Matrix2)
    @. diff = (Matrix1-Matrix2)*dϵ/2
end