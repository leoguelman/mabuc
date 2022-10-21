# mabuc

Multi-arm bandits with unobserved confounders

## Usage

`mabuc` reproduces [Bandits with Unobserved Confounders: A Causal Approach](https://proceedings.neurips.cc/paper/2015/hash/795c7a7a5ec6b460ec00c5841019b9e9-Abstract.html) (Elias Bareinboim, Andrew Forney, Judea Pearl, NIPS 2015).

```python
from mabuc.mabuc import Mabuc
m = Mabuc(pr_D=0.5, pr_B=0.5, T=1000)
m.generate_samples(R=1000)
m.get_stats()
```


## Todo

* Display-ad domain 
* Bayesian 
* Offline policy optimization 

## License

`mabuc` was created by Leo Guelman. It is licensed under the terms of the MIT license.

