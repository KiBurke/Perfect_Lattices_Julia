# Perfect_Lattices_Julia
A tool for computing perfect lattices over number fields in Julia using Oscar.jl

WORK IN PROGRESS

Contains tools for enumerating perfect (in the additive sense) lattices over totally real and CM number fields.

## Setup
All the code is contained in the script 'perfect_lattices.jl', which uses Oscar.jl and Transducers.jl.  If you're new to Julia, here's how you can use it:

Follow the steps outlined in https://www.oscar-system.org/install to install Oscar (and Julia).

In the Julia interface, install Transducers.jl (https://juliafolds.github.io/Transducers.jl/dev/):
```
import Pkg; Pkg.add("Transducers")
```
Next, download perfect_lattice.jl from this github.  Open julia in the folder containing this file and write:
```
include("perfect_lattices.jl")
```
And you're good to go.

## Example
The perfect binary forms over $K=\mathbb{Q}(\sqrt{6})$:

First, we initialise the space of forms:
```
n = 2; #dimension of forms
pol = [-6,0,1]; #coefficients of the defining polynomial of the field
V = V_init(n,pol);
```
Then we do the computation:
```
class = 1; #Steinitz class of lattice - 1 as field has trivial class group
forms = enumerate_perfect_forms(V,class); #a vector of matrices representing all 22 classes of perfect forms
```
Default behaviour is quite verbose; to silence it, pass the flag `verbose = false`:
```
forms = enumerate_perfect_forms(V,class; verbose = false);
```
When $K$ has nontrivial class group, we take the representative lattices $I\oplus {\mathcal{O}_K}^{n-1}$, where the ideals $I$ are representatives of the ideal class group $Cl(K)$ of the ring of integers of $K$.
We encode this as an integer `class`, such that `V.field.ideals[class]` is the ideal $I$ used to make the lattice.  If `V_init` is used to create `V`, then `V.field.ideals` will contain an ideal for each class in $Cl(K)$.

## Notes
This program makes use of multithreading - be sure to open julia with as many threads as you wish: `julia --threads=8`, for example.

`enumerate_perfect_forms` is memory intensive.  With 24Gb of system memory, testing showed the program crashed after storing $~100,000$ binary forms over $\mathbb{Q}(\zeta_8)$.  Large amounts of RAM is needed for long computations.

Currently, due to a memory leak, the function used to compute dual descriptions can only be called $10,000$ times before crashing.  If you believe your field may have more than 10,000 classes of perfect forms, you should use the function `enumerate_perfect_forms_worker` instead as a work-around.  This will uses distributed computing tools to avoid the error, but takes extra time and memory overhead to occasionally set up extra processes.

The functionality for non free lattices (i.e `class != 1`) has not been tested against any known results, as the author couldn't find any published data about non free perfect module lattices (at least, in the additive sense).  The results seem 'reasonable', but there was nothing to match against. 

Until proper documentation is made - for full functionality, see the source code.
