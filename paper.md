---
title: "SGMT: Slot-Guided Modular Transformers for Compositional Reasoning"
authors:
  - name: Jasraj Budigam
    affiliation: '1'
affiliations:
  - index: 1
    name: Indus International School
---

# Summary

SGMT is a new machine-learning model implemented in PyTorch for tasks that require combining known instructions in novel ways (compositional generalization).  It learns to split each input sequence into *slots* (using a Slot Attention mechanism [@locatello2020]) and then routes each slot through one of several small neural modules (experts).  This “slot+module” design biases the model to treat different parts of an input independently, helping it generalize to new combinations of commands.  As described in [the SGMT paper][@fedus2022; @lake2018], SGMT achieves nearly-perfect accuracy on challenging compositional tests (e.g. ≥99% on the SCAN dataset) [oai_citation:0‡file-hs2hfour3umskx3rtwbvjv](file://file-Hs2hfour3uMskX3rTwBvjV#:~:text=We%20address%20compositional%20generalization%20under,results%20rival%20much%20larger%20sparse).  The software provides the full SGMT implementation: model classes, data generators for SCAN-like tasks, and training scripts.  Users can train SGMT on their data or use the provided examples to reproduce the published results.

# Statement of Need

Many standard neural networks struggle to handle novel combinations of learned concepts [@lake2018], a problem known as *systematic compositionality*.  For example, a model that learns “jump around” and “walk after” must generalize to “walk around” and “jump after” — something humans do easily.  Prior research suggests that explicit structure is needed: classical mixture-of-experts models [@jacobs1994] and modern sparse Transformers [@shazeer2017; @fedus2022] allocate computation into specialized sub-modules.  SGMT combines these ideas with *slot-based abstraction*: each input is broken into multiple “slots” (analogous to object-centric embeddings) and each slot is routed through one of several tiny expert networks.  This inductive bias lets SGMT learn near-symbolic subroutines internally [oai_citation:1‡file-hs2hfour3umskx3rtwbvjv](file://file-Hs2hfour3uMskX3rTwBvjV#:~:text=Figure%203%3A%20Slot,of%20a%20command%20to%20separate), which in turn enables outstanding compositional accuracy on tasks like SCAN [oai_citation:2‡file-hs2hfour3umskx3rtwbvjv](file://file-Hs2hfour3uMskX3rTwBvjV#:~:text=Figure%202%3A%20Activation%20counts%20of,module%20serving%20a%20distinct%20subroutine) [oai_citation:3‡file-hs2hfour3umskx3rtwbvjv](file://file-Hs2hfour3uMskX3rTwBvjV#:~:text=Figure%203%3A%20Slot,of%20a%20command%20to%20separate).  

Existing libraries for Transformers (e.g. HuggingFace) do not natively support slot attention or hard expert routing, and previous compositional models lacked open-source PyTorch code.  This software fills that gap by providing a ready-to-use SGMT implementation.  Researchers studying compositionality, modular networks, or efficient Transformers can use it as a foundation. For example, one could adapt SGMT to new domains (e.g. vision-language commands) or experiment with different numbers of slots/modules. The code emphasizes ease of use: after installing dependencies, users can train SGMT on SCAN variants or run inference with just a few lines (see the README and examples).  In short, SGMT’s code addresses the need for a reproducible, well-documented implementation of a state-of-the-art compositional model.

# Acknowledgements

This work was completed by Jasraj Budigam as part of independent research at Indus International School.  The author thanks colleagues and mentors for helpful discussions (if applicable), and the school for providing resources.  No external funding was used.  

# References

Lake, B. M., & Baroni, M. (2018). *Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks*. In *Proc. of ICML* (pp. 4487–4499). DOI: 10.48550/arXiv.1711.00350.

Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., & Kipf, T. (2020). *Object-Centric Learning with Slot Attention*. In *NeurIPS 33*. DOI: 10.48550/arXiv.2006.15055.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. In *ICLR*. DOI: 10.48550/arXiv.1701.06538.

Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity*. *Journal of Machine Learning Research, 23*(120), 1–47. DOI: 10.48550/arXiv.2101.03961.

Jordan, M. I., & Jacobs, R. A. (1994). *Hierarchical mixtures of experts and the EM algorithm*. In: M. Marinaro & P. G. Morasso (Eds.), *International Conference on Artificial Neural Networks (ICANN)*, Lecture Notes in Computer Science (vol. 880, pp. 181–214). Springer. DOI: 10.1007/978-1-4471-2097-1_113.
