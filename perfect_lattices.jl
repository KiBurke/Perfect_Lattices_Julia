using Oscar
using LinearAlgebra
using Printf
import Transducers #For parallelization

#v0.1.1

#Tools for computing perfect lattices over CM/Totally real number fields.  Work in Progress.
#Written for a dev version of Oscar v1.7.0 - due to the use of internal functions in Hecke.jl, compatibility with
#other versions is not guaranteed.

#Special thanks to the authors of Hecke.jl - https://github.com/thofma/Hecke.jl - The code used
#for checking many isometries in succession is a modification of the source code for `is_isometric`
#between two lattices.  Specifically, it is used for `iso_data_init`, `iso_ctx_init`, 
#`my_init_small` and `my_isometry`.

"Collated data of a number field used to compute perfect lattices"
abstract type FieldData end

"Vector space of Humbert forms over a number field"
abstract type HumbertSpace end

"All the data needed to compute perfect lattices in a CM field"
struct CMData <: FieldData
    c::Hecke.RelSimpleNumField{AbsSimpleNumFieldElem} #Relative extension Kc/Kr
    r::AbsSimpleNumField #Kr; the maximal totally real subfield of Kc
    #Embedding from absolute field of Kc:
    map::NumFieldHom{Hecke.RelSimpleNumField{AbsSimpleNumFieldElem}, AbsSimpleNumField, 
        Hecke.MapDataFromNfRel{AbsSimpleNumFieldElem, 
        Hecke.MapDataFromAnticNumberField{AbsSimpleNumFieldElem}}, 
        Hecke.MapDataFromAnticNumberField{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}, 
        Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}
    #Complex conjugation:
    bar::NumFieldHom{Hecke.RelSimpleNumField{AbsSimpleNumFieldElem}, 
        Hecke.RelSimpleNumField{AbsSimpleNumFieldElem}, 
        Hecke.MapDataFromNfRel{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}, 
        Hecke.MapDataFromAnticNumberField{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}}, 
        Hecke.MapDataFromNfRel{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}, 
        Hecke.MapDataFromAnticNumberField{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}}, 
        Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}
    #Field automorphisms of Kc/QQ:
    abs_autos::Vector{NumFieldHom{Hecke.RelSimpleNumField{AbsSimpleNumFieldElem},  
        Hecke.RelSimpleNumField{AbsSimpleNumFieldElem}, 
        Hecke.MapDataFromNfRel{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}, 
        Hecke.MapDataFromAnticNumberField{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}}, 
        Hecke.MapDataFromNfRel{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}, 
        Hecke.MapDataFromAnticNumberField{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}}, 
        Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}}
    basis::Vector{AbsSimpleNumFieldElem}  #Integral basis of Kr
    basis_matrix::QQMatrix #Change of basis from power basis to integral basis of Kr
    cgen::Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem} #an imaginary element of Kc
    #Integral basis of Kc.  Currently also stored in ideal_basis[1]:
    cbasis::Vector{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}
    #Integral bases of representatives of the class group of Kc:   
    ideal_basis::Vector{Vector{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}}
    #Representatives of the class group of Kc, realised in the relative order O_Kc:
    ideals::Vector{Hecke.RelNumFieldOrderIdeal{AbsSimpleNumFieldElem, 
        AbsSimpleNumFieldOrderFractionalIdeal, 
        Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}}
    cbasis_matrix::QQMatrix #Change of basis from power basis to integral basis of Kc
    #[1,a] where a is primitive in Kc/QQ:
    abs_gen::Vector{Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}} 
end

"All the data needed to compute perfect lattices in a totally real number field"
struct RealData <: FieldData
    r::AbsSimpleNumField #the field K
    basis::Vector{AbsSimpleNumFieldElem} #Integral basis of K
    basis_matrix::QQMatrix #Change of basis from power basis to integral basis of K
    #Field automorphisms of K/QQ:
    abs_autos::Vector{NumFieldHom{AbsSimpleNumField, AbsSimpleNumField,
        Hecke.MapDataFromAnticNumberField{AbsSimpleNumFieldElem}, 
        Hecke.MapDataFromAnticNumberField{AbsSimpleNumFieldElem}, AbsSimpleNumFieldElem}}
    abs_gen::Vector{AbsSimpleNumFieldElem} #[1,a] where a is primitive in K/Q
    #Integral bases of representatives of the class group of K:
    ideal_basis::Vector{Vector{AbsSimpleNumFieldElem}}
    ideals::Vector{AbsSimpleNumFieldOrderIdeal} #Representatives of the class group of K
end

"Vector space of Humbert forms over a CM field"
struct ComplexHumbertSpace <: HumbertSpace
    field::CMData #Underlying field's data
    space::Hecke.QuadSpace{QQField, QQMatrix} #The vector space V itself
    dim::Int64 #The dimension of V over RR
end

"Vector space of Humbert forms over a totaly real number field"
struct RealHumbertSpace <: HumbertSpace
    field::RealData #Underlying field's data
    space::Hecke.QuadSpace{QQField, QQMatrix} #The vector space V itself
    dim::Int64 #The dimension of V over RR
end

"Humbert form with *some* data useful for isometries"
struct FormData
    mat::AbstractAlgebra.Generic.MatSpaceElem{T} where T <: NumFieldElem
    scalars::Vector{QQFieldElem}
    Zgram::Vector{ZZMatrix}
end

"Humbert form with *all* data useful for isometries"
struct FormCtx
    mat::AbstractAlgebra.Generic.MatSpaceElem{T} where T <: NumFieldElem
    scalars::Vector{QQFieldElem}
    ctx::Hecke.ZLatAutoCtx
end

"Write `a` in integral coordinates using the basis matrix stored in `K`"
function integral_coordinates(a::NumFieldElem,K::FieldData)
    return(coordinates(K.r(a))*K.basis_matrix)
end

"Grab the biggest field stored in `K` - the totally complex one if `K` is CM, the totally real one if `K` is totally real"
function top_field(K::FieldData)
    if K isa RealData
        return K.r
    else
        return K.c
    end
end

"Makes a symmetric matrix out of `v`"
function symmetric_matrix(v::AbstractVector{T}) where {T <: NCRingElement}
    M = upper_triangular_matrix(v)
    for i in range(1,number_of_rows(M))
        for j in range(i+1,number_of_rows(M))
            M[j,i] = M[i,j]
        end
    end
    return M
end

"Makes an antisymmetric matrix out of `v`"
function antisymmetric_matrix(v::AbstractVector{T}) where {T <: NCRingElement}
    M = strictly_upper_triangular_matrix(v)
    for i in range(1,number_of_rows(M))
        for j in range(i+1,number_of_rows(M))
            M[j,i] = -M[i,j]
        end
    end
    return M
end

"Make a `FieldData` object, given the number field K and fills out all necessary information"
function field_init(K::NumField)
    if is_totally_real(K)
        o = maximal_order(K)
        clgp,clmap = class_group(K)
        clgp = clmap.(clgp)
        idbasis = [elem_in_nf.(basis(I)) for I in clgp]
        b = basis(o,K)
        bm = inv(basis_matrix(o))
        aag = automorphism_list(K)
        g = elem_type(K)[K(1), absolute_primitive_element(K)]
        if aag[1](g[2]) != g[2] #Just checking the first auto is identity pending investigation
            error("First auto was not identity")
        end
        return(RealData(K,b,bm,aag,g,idbasis,clgp))
    elseif Hecke.is_cm_field(K)[1]
        d = degree(K)
        Kr = K
        subfields = [L[1] for L in principal_subfields(K)] #Find maximal totally real subfield
        for field in subfields 
            if degree(field) == d/2 && is_totally_real(field)
                Kr = field #Messy, but only need to call once so whatever
            end
        end
        Kc,m = relative_simple_extension(K,Kr)
        o = maximal_order(Kr)
        b = basis(o,Kr)
        bm = inv(basis_matrix(o))
        ag = automorphism_group(Kc)
        con = ag[2](gens(ag[1])[1])
        g = gen(Kc)-con(gen(Kc))
        oc = maximal_order(K)
        clgp,clmap = class_group(oc)
        clgp = clmap.(clgp)
        orel = maximal_order(Kc)
        ids = [inv(m).(K.(a)) for a in gens.(clgp)]
        ids = [map(X -> ideal(orel,orel(X)), a) for a in ids]
        ids = [reduce(+,a) for a in ids]
        idbasis = [inv(m).(elem_in_nf.(basis(I))) for I in clgp]
        cb = [inv(m)(a) for a in basis(oc,K)]
        cbm = inv(basis_matrix(oc))
        aag = automorphism_list(K)
        if aag[1](gen(K)) != gen(K) #Insurance, can remove after debug
            error("First auto was not identity")
        end
        Kcautos = [compose(m,compose(a,inv(m))) for a in aag]
        cabs = elem_type(Kc)[Kc(1), absolute_primitive_element(Kc)]
        return CMData(Kc,Kr,m,con,Kcautos,b,bm,g,cb,idbasis,ids,cbm,cabs)
    end
    error("Field is neither totally real nor CM")
end

"Make a vector in any Humbert space over totally real `K` into a matrix represnting the form.
    The `n` is there only for compatibility with CM version"
function humbert_to_mat(v::Vector{QQFieldElem}, K::RealData; n = nothing)
    b = K.basis
    d = length(b)
    #Convert to number field elements:
    entries = [sum([b[j]*v[d*(i-1)+j] for j in range(1,d)]) for i in range(1,length(v)÷d)]
    return symmetric_matrix(entries)
end

"Make a vector in any Humbert space over CM `K` into a matrix represnting the form
    Supply `n`, the dimension of V over K tensor RR to avoid computing an unessecary square root"
function humbert_to_mat(v::Vector{QQFieldElem}, K::CMData; n = nothing) 
    b = K.basis
    d = length(b)
    if n === nothing
        n = isqrt(length(v)÷d)
    end
    #Convert to number field elements:
    entries = [sum([b[j]*v[d*(i-1)+j] for j in range(1,d)]) for i in range(1,length(v)÷d)]
    symm = symmetric_matrix(entries[1:(n*(n+1))÷2]) #Real symmetric part
    if n!=1 
        antisymm = antisymmetric_matrix(entries[((n*(n+1))÷2)+1:end]) #Imaginary antisymmetric part
    else
        antisymm = symm*0 #If unary, don't need anything else
    end
    return symm+K.cgen*antisymm
end


"Make a `HumbertSpace` of dimension `n` over K tensor RR
    Totally real version"
function humbert_init(K::RealData, n::Int)
    d = length(K.basis)*(n*(n+1))÷2
    G = zero_matrix(QQ,d,d)
    #Generate inner product matrix for <A,B> = tr(tr(A*B))
    for i in range(1,d)
        vi = [QQ(0) for k in range(1,d)]
        vi[i] = 1
        for j in range(1,d) 
            vj = [QQ(0) for k in range(1,d)]
            vj[j] = 1
            G[i,j] = absolute_tr(tr(humbert_to_mat(vi,K)*humbert_to_mat(vj,K)))
        end
    end
    return RealHumbertSpace(K,quadratic_space(QQ,G),n)
end

"Make a `HumbertSpace` of dimension `n` over K tensor RR
    CM version"
function humbert_init(K::CMData, n::Int)
    d = length(K.basis)*n^2
    G = zero_matrix(QQ,d,d)
    #Generate inner product matrix for <A,B> = tr(tr(A*B))
    for i in range(1,d)
        vi = [QQ(0) for k in range(1,d)]
        vi[i] = 1
        for j in range(1,d) 
            vj = [QQ(0) for k in range(1,d)]
            vj[j] = 1
            G[i,j] = absolute_tr(tr(humbert_to_mat(vi,K,n = n)*humbert_to_mat(vj,K,n = n)))
        end
    end
    return ComplexHumbertSpace(K,quadratic_space(QQ,G),n)
end


"Turn a form represented by a matrix `q` into one represented by a vector in the Humbert space `V`.
    Totally real version"
function mat_to_humbert(q::AbstractAlgebra.Generic.MatSpaceElem{T}, V::RealHumbertSpace
        ) where T <: NumFieldElem
    d = length(V.field.basis)
    counter = 0
    v = [QQ(0) for i in range(1, dim(V.space))]
    n = number_of_rows(q)
    for i in range(1,n)
        for j in range(i,n)
            counter = counter+1
            v[d*(counter-1)+1:d*counter] = integral_coordinates(q[i,j],V.field)
        end
    end
    return v
end

"Turn a form represented by a matrix `q` into one represented by a vector in the Humbert space `V`.
    CM version"
function mat_to_humbert(q::AbstractAlgebra.Generic.MatSpaceElem{T}, V::ComplexHumbertSpace
        ) where T <: NumFieldElem
    d = length(V.field.basis)
    n = number_of_rows(q)
    qbar = map(V.field.bar,q)
    qr = map(X->X/2, q + qbar)
    qc = map(X->X/(2*V.field.cgen), q - qbar)
    counter = 0
    v = [QQ(0) for i in range(1, dim(V.space))]
    for i in range(1,n)
        for j in range(i,n)
            counter = counter+1
            v[d*(counter-1)+1:d*counter] = integral_coordinates(qr[i,j],V.field)
        end
    end
    for i in range(1,n)
        for j in range(i+1,n)
            counter=counter+1
            v[d*(counter-1)+1:d*counter] = integral_coordinates(qc[i,j],V.field)
        end
    end
    return v
end


#TODO speed up traceform using static arrays or something

"Take the 'trace form' of a Humbert form over `K` represented by the matrix `q`.
Set `class` to pick a Steinitz class for your lattice, based on representatives in `K.ideals`.
Totally real version"
function trace_form(q::AbstractAlgebra.Generic.MatSpaceElem{T}, K::RealData; class::Int = 1
        ) where T <: NumFieldElem
    blocks = Vector{Matrix{QQFieldElem}}(undef,number_of_rows(q)^2)
    counter = 0
    d = length(K.basis)
    #Need entries of the form tr(q_i,j*w_i*w_j)
    for k in eachindex(q) #First row/column use the ideal, the rest use a basis of O_K 
        counter = counter+1
        if k[1] == 1 && k[2] == 1
            tracemat = reshape(
                [tr(q[k]*(K.ideal_basis[class][i])*K.ideal_basis[class][j])
                for i in range(1,d) for j in range(1,d)],(d,d))
        elseif k[1] == 1
            tracemat = reshape(
                [tr(q[k]*(K.ideal_basis[class][i])*K.basis[j]) 
                for i in range(1,d) for j in range(1,d)],(d,d))
        elseif k[2] == 1
            tracemat = reshape(
                [tr(q[k]*K.basis[i]*K.ideal_basis[class][j]) 
                for i in range(1,d) for j in range(1,d)],(d,d))
        else
            tracemat = reshape(
                [tr(q[k]*K.basis[i]*K.basis[j]) 
                for i in range(1,d) for j in range(1,d)],(d,d))
        end
        blocks[counter] = tracemat
    end
    #Splatting is slow (I've heard) :(
    return matrix(QQ,d*number_of_rows(q),d*number_of_rows(q),hvcat(number_of_rows(q),blocks...))
end


"Take the 'trace form' of a Humbert form over `K` represented by the matrix `q`.
Set `class` to pick a Steinitz class for your lattice, based on representatives in `K.ideals`.
CM version"
function trace_form(q::AbstractAlgebra.Generic.MatSpaceElem{T}, K::CMData; class::Int = 1
        ) where T <: NumFieldElem
    blocks = Vector{Matrix{QQFieldElem}}(undef,number_of_rows(q)^2)
    counter = 0
    d = length(K.cbasis)
    #Need entries of the form tr(q_i,j*w_i*bar(w_j))
    for k in eachindex(q) #First row/column use the ideal, the rest use a basis of O_K 
        counter = counter+1
        if k[1] == 1 && k[2] == 1
            tracemat = reshape(
                [absolute_tr(q[k]*(K.ideal_basis[class][i])*K.bar(K.ideal_basis[class][j]))
                for i in range(1,d) for j in range(1,d)],(d,d))
        elseif k[1] == 1
            tracemat = reshape(
                [absolute_tr(q[k]*(K.ideal_basis[class][i])*K.bar(K.cbasis[j]))
                for i in range(1,d) for j in range(1,d)],(d,d))
        elseif k[2] == 1
            tracemat = reshape(
                [absolute_tr(q[k]*(K.cbasis[i])*K.bar(K.ideal_basis[class][j]))
                for i in range(1,d) for j in range(1,d)],(d,d))
        else
            tracemat = reshape(
                [absolute_tr(q[k]*(K.cbasis[i])*K.bar(K.cbasis[j]))
                for i in range(1,d) for j in range(1,d)],(d,d))
        end
        blocks[counter] = tracemat
    end
    #Splatting is slow (I've heard) :(
    return matrix(QQ,d*number_of_rows(q),d*number_of_rows(q),hvcat(number_of_rows(q),blocks...))
end



"Takes a trace form `Q` and gives its minima on the `cbasis` of `K`.
    Set `class` to pick a Steinitz class for your lattice, based on representatives in `K.ideals`.
    CM version"
function minimals_in_field(Q::QQMatrix, K::CMData; class::Int = 1)
    sv = shortest_vectors(integer_lattice(gram = Q))
    s = length(sv)
    d = length(K.cbasis)
    Kv = Vector{NTuple{length(sv[1])÷d,Hecke.RelSimpleNumFieldElem{AbsSimpleNumFieldElem}}
        }(undef,s)
    for k in range(1,s) #Picks out first element for basis from ideal
        Kv[k] = Tuple(vcat([sum([K.ideal_basis[class][j]*sv[k][j] for j in range(1,d)])],
            [sum([K.cbasis[j]*sv[k][d*(i-1)+j] for j in range(1,d)]) #Rest from O_K
            for i in range(2,length(sv[1])÷d)]))
    end
    return Kv
end

"Takes a trace form `Q` and gives its minima on the `basis` of `K`.
    Set `class` to pick a Steinitz class for your lattice, based on representatives in `K.ideals`.
    Totally real version"
function minimals_in_field(Q::QQMatrix, K::RealData; class::Int = 1)
    sv = shortest_vectors(integer_lattice(gram = Q))
    s = length(sv)
    d = length(K.basis)
    Kv = Vector{NTuple{length(sv[1])÷d,Hecke.AbsSimpleNumFieldElem}}(undef,s)
    for k in range(1,s) #picks out first element for basis from ideal
        Kv[k] = Tuple(vcat([sum([K.ideal_basis[class][j]*sv[k][j] for j in range(1,d)])],
            [sum([K.basis[j]*sv[k][d*(i-1)+j] for j in range(1,d)]) 
            for i in range(2,length(sv[1])÷d)]))
    end
    return Kv
end

"Turn vector `v` with entries in `K` into the humbert form `v*v`; the 'projection onto `v`'.
    CM version"
function projection_matrix(v::Tuple{Vararg{T}}, K::CMData) where T <: NumFieldElem
    d = length(v)
    projmat = matrix(K.c,d,d,[K.bar(v[i])*v[j] for i in range(1,d) for j in range(1,d)])
    return projmat
end

"Turn vector `v` with entries in `K` into the humbert form `v*v`; the 'projection onto `v`'.
    Totally real version"
function projection_matrix(v::Tuple{Vararg{T}}, K::RealData) where T <: NumFieldElem
    d = length(v)
    projmat = matrix(K.r,d,d,[v[i]*v[j] for i in range(1,d) for j in range(1,d)])
    return projmat
end

"Find the generating rays of the perfect cone over the form in `V` with matrix `q`.
    Set `class` to pick a Steinitz class for your lattice, based on representatives in `K.ideals`"
function perfect_cone_generators(q::AbstractAlgebra.Generic.MatSpaceElem{T}, V::HumbertSpace;
        class::Int = 1) where T <: NumFieldElem
    Q = trace_form(q,V.field; class)
    mins = minimals_in_field(Q,V.field; class)
    minvecs = [mat_to_humbert(projection_matrix(v,V.field),V) for v in mins]
    return unique(minvecs)
end

"Boolean test for whether `q` is perfect in `V`.
    Set `class` to pick a Steinitz class for your lattice, based on representatives in `K.ideals`"
function is_perfect(q::AbstractAlgebra.Generic.MatSpaceElem{T}, V::HumbertSpace; 
        class::Int=1) where T <: NumFieldElem
    b = matrix(QQ,transpose(stack(perfect_cone_generators(q,V; class))))
    W = orthogonal_complement(V.space,b)
    return(is_empty(W))
end

"For perfect trace form `Q` of q with facet trace `F` of f, find the `lambda` such that 
    q+'lambda'*f is perfect.
    `i` is for printing purposes"
function find_lambda(Q::QQMatrix,F::QQMatrix; i::Int = 0)
    #find initial:
    Qf = Symmetric(float(Rational.(Matrix(Q))))
    Ff = Symmetric(float(Rational.(Matrix(F))))
    Ch = cholesky(Qf)
    C = Symmetric(inv(Ch.L)*Ff*inv(Ch.U))
    es = sort(eigenvalues(Float64.(C))) #Potentially low precision
    b = -1/es[1]
    h = (4/3)^((number_of_rows(Q)-1)/2)
    t = (1-(1/2)*(det(Q)*h^number_of_rows(Q)*prod([1+(abs(ei)/abs(es[1])) for ei in es[2:end]]))^(-1))*b
    #Conceivably converting to rational here could cause an issue if it gets too small...
    tr = rationalize(t)
    if tr == 1//0
        tr = Rational(t) #More precision
    end
    G = Q+tr*F
    L = integer_lattice(gram = G)
    v = shortest_vectors(L)[1]
    m = 0 #Always loop once to get an L with smaller denominators than what rationalize produces
    while m < 1//2
        r = (1 - dot(v,Q,v))//dot(v,F,v)
        G = Q+r*F
        L = integer_lattice(gram = G)
        v = shortest_vectors(L)[1]
        m = dot(v,G,v)
    end
    #Avoid anything that would make us divide by 0:
    vs = filter(X->dot(X,F,X)!=0,[u[1] for u in short_vectors(L,1)])
    lams = [(1 - dot(u,Q,u))//dot(u,F,u) for u in vs]
    lambda = minimum(lams)
    if i > 0
        print("\e[u")
        print(i)
    end
    if lambda == 0
        return(r) #We were right all along!
    else
       return(lambda)
    end
end

"Finds a perfect form in V.
    Set `class` to pick a Steinitz class for your lattice, based on representatives in `K.ideals.`
    Set `q` to pick an intitial (non-perfect) from that's not a multiple of Id"
function initial_perfect_form(V::HumbertSpace; q = nothing, class::Int = 1)
    if q === nothing #Construct an intial pos. def form
        q = identity_matrix(top_field(V.field),V.dim)
        q = q/minimum(integer_lattice(gram = trace_form(q,V.field; class)))
    end
    b = matrix(QQ,transpose(stack(perfect_cone_generators(q,V; class))))
    W = orthogonal_complement(V.space,b)
    #Keep finding new forms until its perfect:
    while !is_empty(W)
        #For `f`, we need to make sure to pick something that is not semidefinite.
        #For now, we just sum over the generators, but TODO something more robust
        f = humbert_to_mat(Vector{QQFieldElem}(sum([v for v in eachrow(W)])),V.field,n = V.dim)
        l = find_lambda(trace_form(q,V.field; class),trace_form(f,V.field; class))
        q = q+l*f
        b = matrix(QQ,transpose(stack(perfect_cone_generators(q,V; class))))
        W = orthogonal_complement(V.space,b)
    end
    return(q)
end

"Creates the quadratic/hermitian lattice with gram matrix `q` for isometry purposes.
    Set `class` to pick a Steinitz class for your lattice, based on representatives in `K.ideals`"
function form_to_lattice(q::AbstractAlgebra.Generic.MatSpaceElem{T}, V::HumbertSpace;
        class::Int = 1) where T <: NumFieldElem
    ids = vcat(V.field.ideals[class],repeat([V.field.ideals[1]],number_of_rows(q) - 1))
    pm = pseudo_matrix(identity_matrix(top_field(V.field),number_of_rows(q)),ids)
    if V isa ComplexHumbertSpace
        return hermitian_lattice(V.field.c, pm; gram = q)
    else
        return quadratic_lattice(V.field.r, pm; gram = q)
    end
end

"For testing purposes"
function form_to_lattice_abs(q::AbstractAlgebra.Generic.MatSpaceElem{T}, V::HumbertSpace;
        class::Int = 1) where T <: NumFieldElem
    if V isa ComplexHumbertSpace
        ids = vcat(V.field.abs_ideals[class],repeat([V.field.abs_ideals[1]],number_of_rows(q) - 1))
        pm = pseudo_matrix(identity_matrix(ids[1].order.nf,number_of_rows(q)),ids)
        #ids = vcat(V.field.ideals[class],repeat([V.field.ideals[1]],number_of_rows(q) - 1))
        #pm = pseudo_matrix(identity_matrix(V.field.c,number_of_rows(q)),ids)
        return hermitian_lattice(ids[1].order.nf, pm; gram = V.field.map.(q))
    else
        return quadratic_lattice(V.field.r; gram = q)
    end
end

#This was a nightmare:

"Given a form in `V` with matrix `q`, determine the normals of the cone of `q` up to `aut(q)`.
    Set `class` to pick a Steinitz class for your lattice, based on representatives in `K.ideals.`
    Due to a memory leak in Sympol, this will crash if you call it more than 10,000 times;
    Please use the 'worker' version if that is a concern"
function perfect_cone_normals(q::AbstractAlgebra.Generic.MatSpaceElem{T},V::HumbertSpace;
        class::Int = 1, verbose::Bool = false) where T <: NumFieldElem
    vgens = perfect_cone_generators(q,V; class)
    verbose && @printf " | Rays: %i " length(vgens)
    conegens = [humbert_to_mat(r,V.field) for r in vgens]
    gs = automorphism_group_generators(form_to_lattice(q,V; class))
    if V isa ComplexHumbertSpace #Decide whether involution is needed
        rgs = [[transpose(map(V.field.bar,g))*r*g for r in conegens] for g in gs]
    else
        rgs = [[transpose(g)*r*g for r in conegens] for g in gs]
    end
    #Use rgs to make permutations
    ps = [indexin(conegens,rg) for rg in rgs]
    #GC.enable(false)
    ps = [[x-1 for x in v] for v in ps]
    
    #println("action:")
    pa = Polymake.group.PermutationAction(GENERATORS = ps)
    #println("group:")
    G = Polymake.group.Group(RAYS_ACTION = pa)
    #Very specific typing necessary:
    #genmat = convert(Polymake.LibPolymake.MatrixAllocated{Polymake.LibPolymake.Rational},
        #Polymake.Matrix(permutedims(stack(vgens))))
    #println("mat:")
    genmat = Polymake.Matrix(permutedims(stack(vgens)))

    
    #We must force the input rays to be extremals to get the next function to work.
    #Luckily, they are indeed guaranteed to be extremals
    #println("cone:")
    c = Polymake.polytope.Cone(RAYS = genmat, GROUP = G)
    #This step is leaky; I have informed Polymake hopefully it will get fixed:
    #println("ddp:")
    ntemp = Polymake.polytope.representation_conversion_up_to_symmetry(c)
    #println("normals:")
    normals = eachrow(matrix(QQ,ntemp))
    #println("after:")
    #finalize(normals)
    #GC.enable(true)
    #normals = eachrow(matrix(QQ,Polymake.polytope.representation_conversion_up_to_symmetry(c)))
    #Try clean up to avoid errors:
    #pa = nothing
    #G = nothing
    #genmat = nothing
    #c = nothing
    #ntemp = nothing
    #GC.gc()
    finalize(pa)
    finalize(G)
    finalize(genmat)
    finalize(c)
    finalize(ntemp)
    finalize(normals)


    verbose && @printf(" | Orbits: %i ", length(normals))
    #These normals are w.r.t the standard inner product on R^n.  Here is dirty conversion.
    #Can't nec do a change of basis because the cholesky of V.space.gram has no reason to have rational entries 
    #TODO proper change of base - could maybe scale all the rows separately to make it work?
    #Test which rays are in each facet: 
    my_is_perp = X -> (Y->dot(X,Y)==0)
    indices_in_facets = Vector{Vector{Int64}}(undef,length(normals))
    for i in eachindex(normals)
        indices_in_facets[i] = findall(my_is_perp(normals[i]),vgens)
    end
    #Make the normals with the correct inner product
    fnorms = [orthogonal_complement(V.space,matrix(QQ,transpose(stack(vgens[inds]))))[1,:] 
        for inds in indices_in_facets]
    #Get the right direction:
    for i in eachindex(fnorms)
        j = findfirst(!in(indices_in_facets[i]),eachindex(vgens))
        fnorms[i] = sign(dot(vgens[j],V.space.gram,fnorms[i]))*fnorms[i]

    end
    fnorms = [humbert_to_mat(f,V.field) for f in fnorms]
    #finalize(pa)
    #finalize(G)
    #finalize(genmat)
    #finalize(c)
    #finalize(normals)
    #GC.enable(true)
    return fnorms
end

"Gets the perfect neighbours of form in `V` with matrix `q`.
    Set `class` to pick a Steinitz class for your lattice, based on representatives in `K.ideals`"
function perfect_neighbours(q::AbstractAlgebra.Generic.MatSpaceElem{T},V::HumbertSpace;
        class::Int = 1, verbose::Bool = false) where T <: NumFieldElem
    fs = perfect_cone_normals(q,V; class, verbose)
    verbose && print("| Lambdas: ")
    verbose && print("\e[s")
    verbose && print("...")
    Q = trace_form(q,V.field; class)
    lambdas = Vector{QQFieldElem}(undef,length(fs))
    Threads.@threads for i in eachindex(lambdas)
        lambdas[i] = find_lambda(Q,trace_form(fs[i],V.field; class))
        verbose && @printf("\e[u%i",i)
    end #could also compute the output over threads for small improvement
    verbose && print("\e[u")
    verbose && print(length(lambdas))
    return [q+lambdas[i]*fs[i] for i in eachindex(fs)]
end

"Finds all the perfect forms in `V` for lattices with Steinitz class
    represented by `V.field.ideals[class]`.
    Much quicker than `worker` version, but will crash on the 10,001 cone computed.
    EXPERIMENTAL: Set `input` and `output` files for partial computations: 
    starts with the partial list in `input` (in the format given in a previous `output`),
    and after every form, save to `output`."
function enumerate_perfect_forms(V::HumbertSpace, class::Int = 1;
        input = nothing, output = nothing, verbose::Bool = true)
    if input === nothing
        initial = iso_ctx_init(initial_perfect_form(V; class),V; class) 
        forms = field_conjugates(initial,V; class) #Includes field conjugates
        noconj = [1] #picks representatves of forms up to field automorphism
        counter = 0
    else 
        class, counter, noconj, forms = load_form_data(V,input; iso = true)
    end
    #Keep checking new forms
    while counter < length(noconj)
        counter = counter+1
        verbose && @printf("Forms: %i | Number: %i ", length(noconj), counter)
        newforms = [iso_data_init(q,V; class) for q in 
            perfect_neighbours(forms[noconj[counter]].mat,V; class,verbose)]
        verbose && print(" | Isometry: \e[s")
        for i in eachindex(newforms)
            verbose && @printf("\e[u%i",i)
            #Only consider those with the same scalars:
            filtered = forms |> Transducers.Filter(X->X.scalars == newforms[i].scalars) |> 
                Transducers.Map(X->X.ctx.max)
            #Find the maximum length of short vectors needed:
            bound = Transducers.foldxt(max, filtered; init = 0)
            if bound == 0 #...then we already know it's a new form
                formtoadd = FormCtx(newforms[i].mat, newforms[i].scalars, 
                    (my_init_small(newforms[i].Zgram;)))
                push!(noconj,lastindex(forms)+1) #Add the one form to `noconj`
                #Add all conjugates to the full list:
                forms = vcat(forms,field_conjugates(formtoadd,V; class)) 
            else #...we have to actually check the isometries
                #Sorted to make it easier to truncate:
                shortvectors = (bound, sort(Hecke._short_vectors_gram_integral(Vector, 
                    newforms[i].Zgram[1], bound, Int; is_lll_reduced_known = true), by = last))
                lazyisom = Transducers.Map(X -> my_isometry(newforms[i], X, shortvectors))(forms)
                if !Transducers.foldxt(Transducers.right, Transducers.ReduceIf(identity),
                        lazyisom; init=false) #fancy way of parallel checking
                    formtoadd = FormCtx(newforms[i].mat, newforms[i].scalars,
                        (my_init_small(newforms[i].Zgram; shortvecs=shortvectors)))
                    push!(noconj,lastindex(forms)+1) #Add the one form to `noconj`
                    #Add all conjugates to the full list:
                    forms = vcat(forms,field_conjugates(formtoadd,V; class))
                end
            end
        end
        output !== nothing && save(output,(class,counter,noconj,[p.mat for p in forms]))
        verbose && print("\n")
    end
    return [p.mat for p in forms]
end

"Finds the perfect forms in `V` for each Steinitz class of lattice.
    Only use this for small examples; it's easy to run out of memory with bigger ones.
    For large examples, compute and save to disk one at a time instead.
    `input` and `output` don't work yet"
function enumerate_perfect_forms(V::HumbertSpace)
    #As far as I can tell, list comprehension isn't too slow
    return [enumerate_perfect_forms(V,i) for i in eachindex(V.field.ideals)]
end

"EXPERIMENTAL: Takes file as output by `enumerate_perfect_forms`
    and makes sure its compatible with `V`.  
    If iso is set, then also make isometry data
    CM version"
function load_form_data(V::ComplexHumbertSpace,filename::String; iso::Bool = false)
    class, counter, noconjugates, mats = load(filename)
    #annoying conversion to make mats work with v
    K = V.field
    L = parent(mats[1][1,1])
    absK = codomain(K.map)
    absL, Lmap = absolute_simple_field(L)
    fl, absmap = is_isomorphic_with_map(absL, absK)
    !fl && error("Parent Fields are not Isomorphic")
    relmap = inv(K.map)∘absmap∘inv(Lmap)
    map!(X->relmap.(X), mats)
    if iso
        #print("\e[s")
        conjugates = Vector{FormCtx}(undef,length(mats))
        Threads.@threads for i in eachindex(conjugates)
            conjugates[i] = iso_ctx_init(mats[i],V; class)
            #@printf("\e[u%i",i) 
        end
    else
        conjugates = mats
    end
    return class, counter, noconjugates, conjugates
end

"EXPERIMENTAL: Takes file as output by `enumerate_perfect_forms`
    and makes sure its compatible with `V`.  
    If iso is set, then also make isometry data
    Totally Real version"
function load_form_data(V::RealHumbertSpace,filename::String; iso::Bool = false)
    class, counter, noconjugates, mats = load(filename)
    #Convert to make mats work with `V`:
    K = V.field.r
    L = parent(mats[1][1,1])
    fl, LKmap = is_isomorphic_with_map(L, K)
    map!(X->LKmap.(X), mats)
    if iso
        #print("\e[s")
        conjugates = Vector{FormCtx}(undef,length(mats))
        Threads.@threads for i in eachindex(conjugates)
            conjugates[i] = iso_ctx_init(mats[i],V; class)
            #@printf("\e[u%i",i) 
        end
    else
        conjugates = mats
    end
    return class, counter, noconjugates, conjugates
end

"Get *all* the contextual information needed to find isometries of `q`, as in Hecke.jl"
function iso_ctx_init(q::AbstractAlgebra.Generic.MatSpaceElem{T},V::HumbertSpace;
        class::Int = 1) where T <: NumFieldElem
    #Create the integral forms tr(q), tr(a*q) for primitive a:
    L = form_to_lattice(q,V; class)
    Zgram, scalars, _, _ = Hecke.Zforms(L,V.field.abs_gen)
    #LLL reduce:
    Zgramsmall = copy(Zgram)
    _, Tr = lll_gram_with_transform(Zgram[1])
    Ttr = transpose(Tr)
    for i in 1:length(Zgram)
        Zgramsmall[i] = Tr * Zgram[i] * Ttr
    end
    #Make remaining information:
    ctx = my_init_small(Zgramsmall)
    return FormCtx(q,scalars,ctx)
end

"Produces much of the information needed to check for isometries, as in Hecke.jl.
    `G` is the list of integral gram matrices for a form q"
function my_init_small(G::Vector{ZZMatrix}; shortvecs = nothing)
    Co = Hecke.ZLatAutoCtx(G)
    if shortvecs === nothing
        fl, Cismall = Hecke.try_init_small(Co,false)
    else
        b = maximum(diagonal(Co.G[1]))
        if shortvecs[1] > b #won't accept bigger vectors
            ind = findfirst(>(b)∘last,shortvecs[2]) #check if it stays compiled
            if ind === nothing
                shortvecs = (b,shortvecs[2]) #somtimes there are no vectors of maximum allowed size
            else
                shortvecs = (b,shortvecs[2][1:ind-1])
            end
        end
        fl, Cismall = Hecke.try_init_small(Co,false; known_short_vectors = shortvecs)
    end 
    if !fl
        error("NOT SMALL!!!") #Integer overflow risk :( TODO add ZZRingElem support
    end
    return Cismall
end

"Check if `p` and `q` are isometric, using the method in Hecke.jl.
    `shortvecs` are known short vectors for `p`.
    Note `p` and `q` are in different structures.
    This, along with anything else using Ctx rely on internal Hecke.jl functions; 
    This may break with any update."
function my_isometry(p::FormData,q::FormCtx,
        shortvecs::Tuple{Int64, Vector{Tuple{Vector{Int64}, QQFieldElem}}})
    if p.scalars != q.scalars
        return false
    end
    b = q.ctx.max
    if shortvecs[1] > b #Won't accept bigger vectors, so we must truncate
        ind = findfirst(>(b)∘last,shortvecs[2])
        if ind === nothing
            newshorts = (b,shortvecs[2]) #sometimes there are no vectors of maximum allowed size
        else
            newshorts = (b,shortvecs[2][1:ind-1])
        end
    else
        newshorts = shortvecs
    end
    pC = Hecke.ZLatAutoCtx(p.Zgram)
    fl2, pCs = Hecke.try_init_small(pC, true, ZZRingElem(q.ctx.max), depth = 0, bacher_depth = 0; 
        known_short_vectors = newshorts)
    if !fl2
        error("NOT SMALL!!") #Integer overflow risk :( TODO add ZZRingElem support
    end
    b, _ = Hecke.isometry(q.ctx, pCs)
    return b
end

#This uses the same logic as in `enumerate_perfect_forms`; 
#should probably make a separate function for lists up to isometry:

"Find the non-isometric field conjugates of the form in `V` with Ctx `p`.
    Returns a list of `FormCtx`'s, up to isometry"
function field_conjugates(p::FormCtx,V::HumbertSpace; class::Int = 1)
    formslist = [p]
    if length(V.field.abs_autos) > 1 && class == 1 #Only if there actually are automorphisms
        #Same logic as in `enumerate_perfect_forms`; reducing the list up to isomorphism:
        newforms = [iso_data_init(map(a,p.mat),V;class = class) for a in V.field.abs_autos[2:end]]
        for i in range(1,length(newforms))
            filtered = formslist |> Transducers.Filter(X->X.scalars == newforms[i].scalars) |> 
                Transducers.Map(X->X.ctx.max)
            bound = Transducers.foldxt(max, filtered; init = 0)
            if bound == 0
                formtoadd = FormCtx(newforms[i].mat, newforms[i].scalars,
                    (my_init_small(newforms[i].Zgram;)))
                push!(formslist,formtoadd)
            else
                shorts = (bound, sort(Hecke._short_vectors_gram_integral(Vector,
                    newforms[i].Zgram[1], bound, Int; is_lll_reduced_known = true), by = last))
                lazymap = Transducers.Map(X -> my_isometry(newforms[i], X, shorts))(formslist)
                if !Transducers.foldxt(Transducers.right, 
                        Transducers.ReduceIf(identity), lazymap; init=false)
                    formtoadd = FormCtx(newforms[i].mat, newforms[i].scalars,
                        (my_init_small(newforms[i].Zgram; shortvecs=shorts)))
                    push!(formslist,formtoadd)    
                end
            end
        end
    end
    return formslist
end

"Get *some* of the information needed to find isometries of `q`, as in Hecke.jl"
function iso_data_init(q::AbstractAlgebra.Generic.MatSpaceElem{T},V::HumbertSpace;
        class::Int = 1) where T <: NumFieldElem
    #Create the integral forms tr(q), tr(a*q) for primitive a:
    L = form_to_lattice(q,V; class = class)
    Zgram, scalars, _, _ = Hecke.Zforms(L,V.field.abs_gen)
    #LLL reduce:
    Zgramsmall = copy(Zgram)
    _, Tr = lll_gram_with_transform(Zgram[1])
    Ttr = transpose(Tr)
    for i in 1:length(Zgram)
        Zgramsmall[i] = Tr * Zgram[i] * Ttr
    end
    return FormData(q,scalars,Zgramsmall)
end

"Make a `HumbertSpace` from the field `K` defined by the polynomial with coefficients `coeffs` 
    and the dimension `n` over `K` tensor `RR`"
function V_init(n::Int, coeffs::Vector{Int})
    Qx, x = QQ['x']
    pol = Qx(coeffs)
    J, a = number_field(pol)
    K = field_init(J)
    V = humbert_init(K,n)
    #some polynomials I tested with:
    #i = 75; pol = x^4-i*x^3-6*x^2+i*x+1
    #pol = x^4 - x^3 - 4*x^2 + x + 2 seems doable
    #pol = x^4 - x^3 - 3*x^2 + x + 1
    #pol = x^6+x^5+x^4+x^3+x^2+x+1
    #pol = x^4 - 17*x^2 + 36 #Keeps going forever it seems? Big discriminant
    #pol = x^4 - x^3 + 4*x^2 + 3*x + 9
    #pol = x^4 - 17*x^2 + 36
    return V
end





