# under development may not work until whole pipeline done

# neural-lm
focus on shallow fusion on speech recognition

# TODO
- [x] adaptive softmax for large voca (because pytorch offical implementation can't work with torchscript)
- [ ] onnx support and torchscript
- [ ] gru (should support tie_embedding)
- [ ] gru fusion on wenet runtime ctc prefix beam search
- [ ] transformer-xl with cache
- [ ] transformer-xl with cache to fusion 
- [ ] mwer training when lm fusion 
- [ ] etc
