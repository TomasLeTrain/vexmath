# VexMath
Math library for PROS projects focused on speed. Currently implements: 
* various fast/vectorized pseudo-random number generators
* vectorized versions of common math functions (trig, square root, etc.)
* fast normally-distributed prng generator (modified float-based [implementation of ziggurat algorithm](https://github.com/cd-mcfarland/fast_prng))
* simple Entropy class to generate random numbers heavily based on [veranda](https://github.com/Gavin-Niederman/veranda)

# Credits
* **Gavin Niederman** - Veranda: A rand RNG source for vexide programs
    * [Source code](https://github.com/Gavin-Niederman/veranda)
* **Julien Pommier** - Simple ARM NEON optimized sin, cos, log and exp
    * [Blog](http://gruntthepeon.free.fr/ssemath/neon_mathfun.html)
    * [Source code](http://gruntthepeon.free.fr/ssemath/neon_mathfun.h)
* **Christopher D. McFarland** - Fast PRNG: an Exponentially- and Normally-distributed PseudoRandom Number Generator
    * [Paper](https://arxiv.org/abs/1403.6870)
    * [Source code](https://github.com/cd-mcfarland/fast_prng)
