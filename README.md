<p align="center">
  <h1 align="center">Thai-Cross-CLIP</h1>
  <h3 align="center">Thai CLIP text encoder model trained via Teacher Learning</h3>
</p>

## Overview

This model is a Thai text encoder which is a part of CLIP. This model is trained via Teacher Learning method using OpenAi's CLIP as Teacher

This repo contain only Text encoder which is compatible with ViT-B/32 OpenAi's Image encoder. To use this model as CLIP, you need to clone this repo (text encoder) and OpenAi's CLIP (image encoder). Look furthermore in Tutorial notebook [TBA]

## Demo 

[TBA]

## Pre-trained Models

This Text encoder is trained on 2M Thai captions translated from English by AiResearch's MT model using WangchanBERTa as a pretrained model with an additional linear layer on top.
<br>
<br>
| Name |Model Base|Vision Model| Vision Dimensions|#Parameters|
| ----------------------------------|:-----: |:-----: |:-----: |:-----: |
| [WangchanBERTa ViT-B/32](https://huggingface.co/vikimark/CLIP-MSE-WangchanBerta)| [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)| [OpenAI ViT-B/32](https://github.com/openai/CLIP)| 512 | 106 M |

## Acknowledgements

* [AI Builders](https://github.com/ai-builders/ai-builders.github.io) for providing knowledge and support along the way<br />
* [Multilingual-CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP) for Teacher learning method<br />
* [OpenAI's CLIP](https://github.com/openai/CLIP)<br />
* [AIResearch's translation model](https://airesearch.in.th/releases/machine-translation-models)<br />
