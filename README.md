# Audio Transport: A Generalized Portamento
This repository provides a python implementation of the optimal transport based generalization of audio portamento. The implementation is based on the following paper:

- Trevor Henderson, Justin Solomon,
*Audio Transport: A Generalized Portamento via Optimal Transport*, 2019.  [[arXiv]](https://arxiv.org/abs/1906.06763) [[github]](https://github.com/sportdeath/audio_transport)


This code implements the the static effect, taking two audio files as input and combining the two using the effect to produce and output audio file. It combines (or concatenates) the files by interpolating between their spectra using displacement interpolation, a technique that has it's roots in optimal transport. The resulting signal should be a "glide", or portamento, between the two signals.  Examples of generated spectograms can be seen in `audio_test.ipynb`.
We first test the algorithm on a simple gaussian case before extending it to audio files. We also explored the use of unbalanced transport as to be able to deal with files without the same amount of mass.
The horizontal and vertical coherence corrections suggested by Henderson and Solomun were not fully implemented (this is left as future work). Therefore, some of the generated audio signals can sound a little strange and many artifacts can be perceived. Still, a visual validation of the generated audio spectogram can be done.
