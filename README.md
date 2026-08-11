# Model free agent 'planning' and ego centrism

This repo contains the work in progress code and artifacts for the thesis in progress, building off '_Inverting the Bellman Equation: From Q-Values to World Models_' by Letcher et al which demonstrated a method to recover the inferred world model from goal conditioned model free agent Q tables. Specifically I am interested to see how the agent's extracted transition probabilities (also referred to as 'internal world model' or 'inferred world model' or 'internal model') can be used to accurately indicate where an agent will be at a certain step in time (explainability) and how the internal models of different models trained in the same environment can be used to determine which one is more likely to generalise well through comparing policy effectiveness on other internal models.

This readme will evolve with the thesis progress.

## Current progress:
- Literature review complete
- Experimental methodology drafted
  - Prelim phase. Create custom four rooms environment and Bellman inversion.
  - Phase 1. Extract inferred world model and probabalistic trajectories from goal conditioned agents and compare to live environment trajectories.
  - Phase 2. Create multiple distinct goal conditioned agents and contrast agent policy performance on own model vs other agent models. The theory is an agent will perform well on trained goals using its own internal model and more generally, on internal models that are well representative of the environment. It will perform poorly on internal models that are only locally representative of the environment. 
  - Phase 3. Extend the work above through complex custom four rooms environment (where each room has significantly different transition dynamics) and training of local vs globally goal conditioned models. The purpose of this is to identify metrics to compare models for generalisability. This should then be verified through generalisation on unseen goals.

### Prelim phase
Custom four rooms environment based on paper's implementation complete including deterministic and windy variants.

Bellman extraction for both deterministic (special case) and stochastic kernals complete

### Phase 1
Preliminary work complete:
- Goal conditioned Q learning implemented
- World model extraction functional
- Step wise probability distribution (Monte Carlo) implemented using agent policy and extracted world model.
- Probability distribution comparison between agent world model simulated trajectories and live agent in environment trajectories.
- Probability distributions visualised (both world model alone and world model compared to live agent trajectories) with example live trajectory overlaid

**Next steps:** Quantify probability distribution delta. This will likely include KL Divergence between the world model estimate and both the live trajectory samples and the real underlying probability distribution. This will also likely be compared to the KL Divergence between live trajectories and the underlying true distribution as the reference distribution, and potentially a random sample and second live trajectory run against the original live trajectory as the reference distribution. This will allow us to compare the relative divergence between distributions.

Once we have the structure for the above complete, the robust next step is to run this approach for multiple seeds to ensure we get a broad sample

## Current results

Phase 1 is progressing well and it is fascinating seeing the probability distributions unfold. Preliminary experimenting shows that we sometimes end up with key differences between inferred trajectories and live trajectories; this is generally mitigated by increased training but can also be indicative of poor goal spread. More generally this is an aspect which we look much further into, particularly in Phase 3

This image is an example of the resulting probabiilty distribution from 100000 trajectories in both the inferred model (P hat) and the live agent in the environment.

<img width="1193" height="536" alt="comparison" src="https://github.com/user-attachments/assets/0b1836f4-27e0-4a3f-ab29-794e8a1a8d1a" />

