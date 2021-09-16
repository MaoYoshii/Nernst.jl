using LinearAlgebra
using Plots
using Plots.PlotMeasures
using LaTeXStrings
using ColorSchemes

export  eigval_img,
        eigval_img_2d,
        berry_img,
        nernst_img,
        T_img,
        μ_img,
        Tμ_img



"""
    eigval_img(d,M,Ham!)
    
    plot eigen values. You specify the plot area by `k`
"""

function eigval_img(k :: AbstractArray ,M,Ham!)
    plt = Array{Float64,3}(undef,M,length(k),length(k))
    @inbounds Threads.@threads for x in eachindex(k)
        H1 = zeros(ComplexF64,M,M)
        @inbounds for y in eachindex(k)
            Ham!(k[x],k[y],H1)
            plt[:,x,y] = eigvals!(H1)
        end
    end
    p1 = plot3d(fmt=:png)
    for i in 1:M
        plot3d!(k,k, plt[i,:,:] , st = :surface)
    end
    p2 = plot(k,plt[:,:,length(k)÷2+1]', xlabel = L"k_x")
    p3 = plot(k,plt[:,length(k)÷2+1,:]', xlabel = L"k_y")
    plot(
        p1,p2,p3,layout = @layout([a [b ; c]]),size = (900,500)
    )
end

"""
    eigval_img(d,M,Ham!)
    
    plot eigen values. You specify the span of `k` with d
"""

function eigval_img(d :: Int,M,Ham!)
    k = -1:1/(1<<d):1 
    plt = Array{Float64,3}(undef,M,length(k),length(k))
    @inbounds Threads.@threads for x in eachindex(k)
        H1 = zeros(ComplexF64,M,M)
        @inbounds for y in eachindex(k)
            Ham!(k[x],k[y],H1)
            plt[:,x,y] = eigvals!(H1)
        end
    end
    p1 = plot3d(fmt=:png)
    for i in 1:M
        plot3d!(k,k, plt[i,:,:] , st = :surface)
    end
    p2 = plot(k,plt[:,:,1<<d+1]', xlabel = L"k_x")
    p3 = plot(k,plt[:,1<<d+1,:]', xlabel = L"k_y")
    plot(
        p1,p2,p3,layout = @layout([a [b ; c]]),size = (900,500)
    )
end

function eigval_img_2d(k :: AbstractArray ,M,Ham!)
    plt = Array{Float64,3}(undef,M,length(k),length(k))
    @inbounds Threads.@threads for x in eachindex(k)
        H1 = zeros(ComplexF64,M,M)
        @inbounds for y in eachindex(k)
            Ham!(k[x],k[y],H1)
            plt[:,x,y] = eigvals!(H1)
        end
    end
    
    plot(
        [heatmap(k,k,plt[n,:,:]') for n in 1:M]...,size = (900,500)
    )
end

function eigval_img_2d(d :: Int ,M,Ham!)
    k = -1:1/(1<<d):1 
    plt = Array{Float64,3}(undef,M,length(k),length(k))
    @inbounds Threads.@threads for x in eachindex(k)
        H1 = zeros(ComplexF64,M,M)
        @inbounds for y in eachindex(k)
            Ham!(k[x],k[y],H1)
            plt[:,x,y] = eigvals!(H1)
        end
    end
    
    plot(
        [heatmap(k,k,plt[n,:,:]') for n in 1:M]...,size = (900,500)
    )
end

function berry_img(d :: Int,M,Ham!)
    k = -1:1/(1<<d):1
    plt = Array{Float64,3}(undef,M,length(k),length(k))
    @inbounds Threads.@threads for x in eachindex(k)
        H1, H2, H3 = zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M)
        @inbounds for y in eachindex(k)
             plt[:,x,y]=Berry(k[x],k[y],Ham!,H1,H2,H3)
        end
    end
    
    @. plt /= 1<<(2d)
    
    plot(
        [ heatmap(k,k,plt[i,:,:]', title = "band $(i)" , c = cgrad(:bwr)) for i in 1:M]...,
        xlabel = L"k_x" , ylabel = L"k_y",titlefontsize=8
    )    
end

function berry_img(k :: AbstractArray ,M,Ham!)
    plt = Array{Float64,3}(undef,M,length(k),length(k))
    dμ = step(k)^2
    @inbounds Threads.@threads for x in eachindex(k)
        H1, H2, H3 = zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M)
        @inbounds for y in eachindex(k)
             plt[:,x,y]=Berry(k[x],k[y],Ham!,H1,H2,H3)
        end
    end
    
    plt *= dμ
    plot(
        [ heatmap(k,k,plt[i,:,:]', title = "band $(i) " , c = cgrad(:bwr)) for i in 1:M]...,
        xlabel = L"k_x" , ylabel = L"k_y",titlefontsize=8
    )    
end
"""
    nernst_img(d :: Int,T,μ,M,Ham!)
    nernst_img(k :: AbstractArray ,T,μ,M,Ham!)
"""

function nernst_img(d :: Int,T,μ,M,Ham!)
    k = -1:1/(1<<d):1
    plt = Array{Float64}(undef,length(k),length(k))
    @inbounds Threads.@threads for x in eachindex(k)
        H1, H2, H3 = zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M)
        @inbounds for y in eachindex(k)
             plt[x,y]=nernst(k[x],k[y],T,μ,Ham!,H1,H2,H3)
        end
    end
    @. plt /=1<<(2d) # dxdy
    println(sum(plt))
    heatmap(k,k,plt[:,:]',
        xlabel=L"k_x",
        ylabel=L"k_y"
    )
end

function nernst_img(k :: AbstractArray ,T,μ,M,Ham!)
    plt = Array{Float64}(undef,length(k),length(k))
    dμ = step(k)^2
    @inbounds Threads.@threads for x in eachindex(k)
        H1, H2, H3 = zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M)
        @inbounds for y in eachindex(k)
             plt[x,y]=nernst(k[x],k[y],T,μ,Ham!,H1,H2,H3)
        end
    end
    @. plt *= dμ
    println(sum(plt))

    heatmap(k,k,plt[:,:]',
        xlabel=L"k_x",
        ylabel=L"k_y"
    )
end

"""
    T_img(μ,M,Ham! ; rangeT = 50:50:500 , k = -1:1/(1<<4):1)
"""

function T_img(μ,M,Ham! ; rangeT = 50:50:500 , k = -1:1/(1<<4):1)
    plt = zeros(length(rangeT))
    dμ = step(k)^2
    @inbounds for mu in μ
        @. plt = 0.0
        @inbounds Threads.@threads for T in eachindex(rangeT)
            H1, H2, H3 = zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M)
            @inbounds for x in eachindex(k)
                @inbounds for y in eachindex(k)
                    plt[T] += nernst(k[x],k[y],rangeT[T],mu,Ham!,H1,H2,H3)
                end
            end
        end
        plot!(rangeT,plt.*dμ,label= L"\mu = %$(mu)",xlabel = "logT",ylabel="α",marker=true,ms=4,msw=0)
    end
    plot!()
end

"""
    μ_img(T,M,Ham!;rangeμ = -5:0.5:5 ,k = -1:1/(1<<4):1)
"""
function μ_img(T,M,Ham!;rangeμ = -5:0.5:5 ,k = -1:1/(1<<4):1)
    plt = zeros(length(rangeμ))
    dμ = step(k)^2
    @inbounds Threads.@threads for μ in eachindex(rangeμ)
        H1, H2, H3 = zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M)
        @inbounds for x in eachindex(k)
            @inbounds for y in eachindex(k)
                plt[μ] += nernst(k[x],k[y],T,rangeμ[μ],Ham!,H1,H2,H3)
            end
        end
    end
    plot!(rangeμ,plt.*dμ,label= L"T = %$(T)",xlabel = "μ",ylabel="α",marker=true,ms=4,msw=0)
end

"""
    Tμ_img(M,Ham!; rangeT = 100:100:1000 , rangeμ = -5:0.5:5 , k = -1:1/(1<<4):1)
"""

function Tμ_img(M,Ham!; rangeT = 100:100:1000 , rangeμ = -5:0.5:5 , k = -1:1/(1<<4):1)
    plt = zeros(length(rangeT),length(rangeμ))
    dμ = step(k)^2
    @inbounds Threads.@threads for T in eachindex(rangeT)
        H1, H2, H3 = zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M) , zeros(ComplexF64,M,M)
        @inbounds for μ in eachindex(rangeμ)
            @inbounds for x in eachindex(k)
                @inbounds for y in eachindex(k)
                    plt[T,μ] += nernst(k[x],k[y],rangeT[T],rangeμ[μ],Ham!,H1,H2,H3)
                end
            end
        end
    end
    @. plt *= dμ
    heatmap(rangeT,rangeμ,plt',
        c=cgrad(:bwr),
        xlabel = "T",ylabel="μ"
    )
end