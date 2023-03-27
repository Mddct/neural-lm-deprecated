# (deprecated, will reimplement by jax) under development may not work until whole pipeline done

# neural-lm
focus on fusion on speech recognition

# Note 
> When a language model is used wide beam searches often yield
> incomplete transcripts. With narrow beams, the problem is less
> visible due to implicit hypothesis pruning.

See if it appears in ctc+lm fusion

# TODO
- [x] adaptive softmax for large voca (because pytorch offical implementation can't work with torchscript)
- [ ] onnx support and torchscript
- [x] gru
- [x] rnn tie embedding
- [ ] gru fusion on wenet runtime ctc prefix beam search
- [ ] transformer-xl with cache
- [ ] transformer-xl with cache to fusion 
- [ ] mwer training when lm fusion 
- [ ] etc

# reference
- [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/pdf/1412.5567.pdf) 
- [END-TO-END ATTENTION-BASED LARGE VOCABULARY SPEECH RECOGNITION](https://arxiv.org/pdf/1508.04395.pdf)
- [On Using Monolingual Corpora in Neural Machine Translation](https://arxiv.org/pdf/1503.03535.pdf)
- [First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs](https://arxiv.org/pdf/1408.2873.pdf)
- [Towards better decoding and language model integration in sequence to sequence models](https://arxiv.org/pdf/1612.02695.pdf)
- [END-TO-END ATTENTION-BASED LARGE VOCABULARY SPEECH RECOGNITION](https://arxiv.org/pdf/1508.04395.pdf)
- [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309.pdf)
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- [Minimum Word Error Rate Training with Language Model Fusion for End-to-End Speech Recognition](https://arxiv.org/pdf/2106.02302.pdf)
- etc
