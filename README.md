# An Exploration of Mimic Architectures for Residual Network Based Spectral Mapping

We use a residual network to do spectral mapping for speech enhancement
and robust speech recognition. Our architecture works better than
ResNet, and is further improved by using mimic loss.

This work was accepted to SLT under the title listed above. The paper can be found here:
[https://arxiv.org/pdf/1809.09756](https://arxiv.org/pdf/1809.09756)

## Brief instructions

To pre-train the ResNet senone classifier model, use the `train_resnet_critic.py` file
(the parameters are listed in the file). The format of the clean speech data should be
a Kaldi `.ark` file with spectrogram features (plus a Kaldi `.scp` file).
The format of the senone labels should be a text file with numeric indexes.

To pre-train the ResNet spectral mapper model, use the `train_resnet.py` file, which also
has the parameters listed in the file. The noisy and clean speech data should both be
in Kaldi `.ark` files, with corresponding `.scp` files.

To train the mapper model using mimic loss, use the `train_actor_critic_renset.py` file.

Once the model is trained, you can use it to generate cleaned spectrograms, which may
be useful for downstream tasks such as ASR.
