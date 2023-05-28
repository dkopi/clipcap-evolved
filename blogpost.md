# ClipCap Evolved

*28 Mar 2023* | *Authors: Dawid, Abishek, Dheeraj, Priyakshi, Tom*

![](images/coco_1.png)

Image captioning is a challenging vision-language task that involves automatically generating relevant and valid captions that describes the content of an image. 

Over the years, various approaches have been proposed to tackle this task, with different architectures and training methods being employed. Traditionally, most image captioning pipelines rely on a combination of a visual encoder to encode visual information and a textual decoder that generates captions based on the encoded features. Earlier deep learning based image captioning approaches typically used CNN-encoders and RNN-language models ([Karpathy et al.](https://arxiv.org/abs/1412.2306), [Vinyals et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)), however with recent trends [Transformer](https://arxiv.org/abs/1706.03762)-based models have gained popularity. The visual feature extraction stage has also seen significant changes, moving towards the use of multi-modal architectures trained on large-scale data with language supervision, as seen in models like [CLIP (Contrastive Language-Image Pretraining)](https://arxiv.org/abs/2103.00020).  

One prevalent approach to image captioning involves utilizing pretrained vision and language models, which are then fine-tuned. Our baseline approach, called [ClipCap](https://arxiv.org/abs/2111.09734), adheres to this paradigm by employing CLIP-ViT as the visual encoder and GPT-2 as the textual decoder. In their approach, the image embeddings are sent as prefixes of captions to the Language Model (LM) which then generates the next token. Alternative approaches like [Flamingo](https://arxiv.org/abs/2204.14198) and [VC-GPT](https://arxiv.org/abs/2201.12723) fuse visual information from the encoder directly into the layers of a pre-trained LM using a cross attention mechanism.

In this blog post, we share our work building upon ClipCap and address key research questions. We review the background and key components, including models and fine-tuning techniques. Our proposed methods are presented, highlighting improvements over the baseline. We also discuss our experiments, results, and future directions.


## Introducing ClipCap

The authours of ClipCap propose a simple yet effective technique to generate captions. As mentioned before, CLIP is utilised to extract the visual embeddings of the image, which is the condensed representation of the content. This is used as a prefix to the GPT2 input, which then generates the caption based on both the image and the prefix. A simple mapping network is employed to transform the embedding into a compatible format for GPT2. They follow two approaches,

1. Using an MLP mapper, alongwith finetuning GPT2
2. Using a transformer mapper

Their second approach demonstrate that training the mapping network alone can yield competent captioning results while keeping CLIP and the LM frozen.

At the time of their publication, this method achieved comparable performance to State-Of-The-Art approaches on challenging datasets such as Conceptual Captions and nocaps, while being simpler, faster, and lighter. However, it is worth noting a couple of potential weaknesses. Firstly, they failed to explore the utility of unpooled visual representations, which may affect its ability to capture fine-grained visual details; and the limited evaluation with different language models, which may leave room for further exploration and analysis.

<!-- First, ClipCap does not utilize unpooled CLIP-ViT representations, which may limit its ability to capture fine-grained visual details. Additionally, the approach has not been extensively tested with different language models, leaving room for further exploration and analysis.

<!-- Motivation fused -->

<!-- We wanted to extend this idea by using different language models and different ways of fusing the visual and textual features. Specifically, we used [FLAN-T5](https://huggingface.co/flax-community/flant5), a fined-tuned version of the original pretrained encoder-decoder T5 architecture introduced in 2019. We also experimented with using shared MLPs on unpooled representations from CLIP-ViT, which can potentially preserve more information from the image patches. -->


## Background and Key Components

In this section, we introduce the essential models and methods that serve as building blocks for our image captioning architectures.


### CLIP-ViT

Contrastive Language Pre-Training(CLIP) is an efficient method of learning  from natural language supervision developed by OpenAI. Designed to understand and generate meaningful associations between text and images, CLIP models are effective multimodal vision language models that can be used for a range of tasks, including zero-shot image classification and image-text similarity. 

#### Architecture
CLIP architecture consists of two main components, a text encoder, and an image encoder. These two encoders are jointly trained using a contrastive learning approach to predict the correct pairings of a batch of training (image, text) examples. The CLIP model encodes textual and visual information into a multimodal embedding space, with an aim to increase the cosine similarity score of images and text representations. 

<img src="https://s3.hedgedoc.org/demo/uploads/0ce99e57-bf68-4568-ba2c-f31f558513f8.jpeg" alt= “” width="70%" height="40%">


The original clip implementation uses a transformer as its text encoder. For the image encoder, the authors propose two separate architectures, one with a [ResNet](https://arxiv.org/abs/1512.03385), and the other with a [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929).

###### Vision Transformer
The Vision Transformer (ViT) model architecture was introduced in the paper titled “An Image is Worth 16*16 Words: Transformers for Image Recognition at Scale”, where the authors utilise the transformer architecture for image processing tasks. The proposed architecture involves processing images by splitting an image into fixed size patches, linearly embedding them along with positional embeddings, and then inputting the resultant sequence of vectors to a standard transformer architecture.

<img src="https://s3.hedgedoc.org/demo/uploads/231dff94-f16f-40cf-8d91-4f393bddb3af.jpeg" alt= “” width="70%" height="40%">


The results of the experiments demonstrate that the ViT encoder architecture performs better than the ResNet based encoder architecture on a wide range of datasets. Additionally, the baseline ClipCap implementation uses the CLIP-ViT as its image encoder.

In case of CLIP-ViT, the output tokens from the Vision Transformer are pooled into a single vector and passed through a projecting linear layer.



### GPT-2


OpenAI's [GPT-2 (Generative Pretrained Transformer 2)](https://openai.com/research/gpt-2-1-5b-release) is a large transformer-based language model pretrained on an extensive corpus of English text in a self-supervised manner, enabling it to learn a comprehensive understanding of language and generate coherent and contextually relevant text.

GPT-2 is pretrained in a self-supervised way on raw data without any human labelling with an automatic process to generate inputs and labels from those texts. More specifically, the model is trained to predict the next token in sentences. The inputs to the model are sequences of continuous text, with a specific length, and the targets are the same sequences, but shifted one token to the right. The model internally employs a masked self-attention mechanism, ensuring that predictions for a given token only use the inputs up to that token and not the future tokens. This autoregressive training setup enables the model to capture the sequential dependencies and learn the underlying patterns in the language.

There are several sizes of GPT-2 available:

| GPT-2 variant| Small |   Medium | Large | Extra Large |
|--------------|-------|----------|-------|-------------|
|Parameters    | 117M  |  345M    | 762M  |  1,542M     |

### FLAN-T5

[Flan-T5](https://arxiv.org/abs/2210.11416) is a fine-tuned version the original [T5 architecture](https://arxiv.org/abs/1910.10683) intoduced by Google in 2019. 

#### T5 (Text-to-Text Transfer Transformer)

The T5 architecture is built on standard encoder-decoder transformer architecture leveraging an attention mechanism to process sequential data efficiently. Both the encoder and decoder consist of blocks made of self-attention and a feed-forward network, while decoder also contains cross-attention layers. Notably, the decoder's self-attention mechanism also employs an autoregressive or causal self-attention, allowing the model to attend only to past outputs during decoding.

The encoder processes the input text, generating a representation that captures contextual information and semantic understanding. This representation serves as a conditioning signal for the decoder. The decoder, in turn, attends to the conditioned representation and generates the output text step by step, incorporating the conditional signal at each decoding step.

During pre-training, T5 undergoes training on a large corpus of unlabeled text data to learn generalizable knowledge. T5 takes a "text-to-text" approach, where all NLP tasks are formulated as text-to-text problems. This formulation allows T5, once pre-trained on a large corpus, to be fine-tuned on multiple downstream tasks with minimal modifications, making it highly adaptable and efficient. 

#### Flan-T5 (Fine-tuned Language Net)

With the same number of parameters as T5, Flan-T5 takes the solid foundation of T5 and enhances it further by fine-tuning it on over 1000 additional tasks, expanding its coverage of languages and task domains.

Flan-T5 model offers 5 different variants:

|FLAN-T5 variant| Small |   Base | Large | XL | XXL   |
|---------------|-------|--------|-------|----|-------|
|Parameters     | 80M   | 250M   |780M   | 3B | 11B   |



### Parameter Efficient Fine Tuning

A common practice in the field of NLP is the usage of pretrained models and adapting it to other downstream tasks by finetuning it for a particular task or dataset. However, as LLMs are becoming increasingly large with billions of parameters, it becomes prohibitively hard to train such models. Parameter-Efficient Fine-Tuning(PEFT) techniques help overcome this by freezing most of the pretrained model’s parameters and modifying a small subset of the parameters. [“Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Model”](https://arxiv.org/abs/2203.06904) broadly classifies these techniques into 3 categories:

 - Addition-based methods, which add new parameters to the original model and fine-tune them while keeping the original parameters fixed. Examples include [Adapters-based tuning](http://proceedings.mlr.press/v97/houlsby19a.html) and [prompt based tuning](https://arxiv.org/abs/2108.02035).
 - Specification-based methods only fine-tune a subset of the original parameters that are specified by [heuristics](https://arxiv.org/abs/1911.03090).
 - Reparameterization-based methods transform the adaptive parameters during optimization into parameter efficient forms, typically motivated by the hypothesis that LM adaptations towards most downstream tasks are inherently low-rank, and could thus be equivalently completed in a parameter-efficient way. A particularly interesting implementation is that of [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) of large language models. 

#### Low Rank Adaptation

Low Rank Adaptation (LoRA) is a technique introduced by Hu et al. in their paper as efficient fine tuning technique that can greatly reduce the number of trainable parameters for downstream tasks, by freezing the pre-trained model weights and injecting trainable rank decomposition matrices into each layer of the Transformer architecture. In particular, LoRA use tensor-train decomposition of the weight matrix and decompose it into two "update matrices".

<img src="https://miro.medium.com/v2/resize:fit:730/1*D_i25E9dTd_5HMa45zITSg.png" alt= “” width="50%" height="50%">

For any model layer that can be expressed as a matrix multiplication of the form $h=W_0x$, it can be reparametrised as follows 

$h = W_0+\frac{\alpha}{r}BAx$

where, $A\in\mathbb{R}^{r \times k}$ and $B\in\mathbb{R}^{d \times r}$ and *r* is the low dimensional rank of the decomposition. 

##### Advantages of LoRA

- Since the original pretrained language model is frozen, training using LoRA is more efficient, as we do not need to calculate the gradients or maintain optimizer states for most parameters, and calculate only for the injected much smaller lower rank matrices. The authors note that for a large Transformer trained with Adam, we reduce that VRAM usage by up to 2/3 if r <<  $d_{model}$ as we do not need to store the optimizer states for the frozen parameters
- Adapter layers often introduce inference latency, by extending model depth or reducing the model’s usable sequence length. LoRA overcomes this issue by proposing that when deploying models in production we can explicitly compute $W = W_0+BA$ and store the weight matrix and perform inference as usual. When the pretrained LM needs to be adapted to a new downstream task, the original $W_0$ can be recovered by subtracting BA and a new B’A’ can be summed for the new task
- LoRA is orthogonal to many prior methods and can be combined with many of them, such as prefix-tuning.


## Approaches & Architectures

In this study, we explore different approaches to improve the generation of accurate and descriptive captions for images, while considering the impact on trainable parameters.

### Baselines

We begin by replicating the ClipCap architectures to establish baseline performance. Our focus lies on replicating two variants of the architecture, while employing the CLIP-VIT/32 model as the visual encoder. These serve as reference points to evaluate and compare the effectiveness of our approaches. 
 
##### ClipCap, The MLP Approach
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   ![](https://)![](https://s3.hedgedoc.org/demo/uploads/1c551205-eed4-4b2d-936a-6bc151b994e1.png)

##### ClipCap, The Transformer Approach
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   ![](https://)![](https://s3.hedgedoc.org/demo/uploads/b456699a-d55a-4fae-87b6-be14c642851a.png)


### Language Model Architecture Matters

ClipCap utilized GPT2, a decoder-only transformer, to generate tokens autoregressively. While this approach proved effective, we sought to optimize it further by introducing an additional conditioning signal to the cross-attention layers. This insight was inspired by techniques used in the Flamingo paper. The addition of a conditioning signal to the decoder blocks is hypothesized to enhance caption generation performance. This signal delivers an additional layer of context at each decoding step, thus facilitating the decoder to construct more accurate and coherent output.

Based on this observation, we explore the use of encoder-decoder models as a promising direction. This resulted in our incorporation of the **Flan-T5** model into the ClipCap architecture. The decision to integrate Flan-T5 into ClipCap was motivated by its versatility in handling a multitude of tasks, each one encoded by the encoder. This presents a unique opportunity for improving the caption prediction process. By feeding a prefixed sentence to the encoder block, we are priming the decoder, theoretically enabling it to predict captions more effectively. This is predicated on the hypothesis that the encoder's capacity to embed different tasks will substantially enhance the decoder's proficiency in generating precise and pertinent captions.

#### Using FLAN-T5 as the LM
![CLIP-VIT/32 + MLP + FLAN-T5](https://s3.hedgedoc.org/demo/uploads/eb688bed-ae5f-47e0-9a9e-96d8d4e9d36d.png)

Here, we chose to examine only the approach using an MLP mapper, given that the transformer from ClipCap inherently serves as an encoder. Introducing a transformer mapper into an already existing encoder-decoder architecture would be redundant and potentially unnecessary.

#### Using FLAN-T5 Decoder Only

Another approach in our exploration involves utilising only the decoder component of the FLAN-T5 model. In this variant, we decided to bypass the encoder and feed the inputs from the previous components directly to the pre-trained cross attention layers of the decoder. We tested this variant with the two mappers: MLP and Transformer.

##### FLAN-T5 Decoder, The MLP Approach
![CLIP-VIT/32 + MLP + FLAN-T5 Decoder Only](https://s3.hedgedoc.org/demo/uploads/763e7342-f649-4dae-a6aa-8a7b42aa77bc.png)

##### FLAN-T5 Decoder, The Transformer Approach
![CLIP-VIT/32 + Transformer + FLAN-T5 Decoder Only](https://s3.hedgedoc.org/demo/uploads/93919a7e-4d0b-4d03-ae75-dc606392ed1c.png)


### Beyond Pooled Features: Exploiting Visual Representations

In order to enhance the utilization of visual representations in our models, we propose a departure from using pooled and projected features. Instead, we advocate for leveraging the unpooled representations, which capture more comprehensive visual information. By preserving the richness of visual details that can be lost through pooling and projection, we aim to provide the language model with a more robust and nuanced image representation.

To effectively incorporate these unpooled visual tokens into our models, we take steps to align the representation spaces of the visual and language models. This involves passing the visual tokens through a Multilayer Perceptron (MLP) with shared weights for all tokens. The MLP serves the purpose of retaining valuable information from each visual token, ensuring that no useful details are overlooked. Subsequently, these refined tokens are fed into the language model. For GPT2 and Flan-T5, they act as the prefix, while for the Flan-T5 decoder, they serve as the entire conditioning signal. We anticipate that this tweak will result in improved performance.

![CLIP-Projections](https://s3.hedgedoc.org/demo/uploads/73177a2b-9659-41d3-9fea-483551cba252.png)

The shared MLP projects the visual tokens that are not pooled. This means we utilise all the tokens that CLIP-ViT outputs. These tokens, after projection, are directly mapped to the LM.

### Parameter-Efficient Fine-tuning with LoRA


Translating between the representations of image encoder (CLIP) and the language model was a challenge faced by the authours of ClipCap. This is owed to their independent training, leading to separate latent spaces. To address this, the authors emphasize the need for fine-tuning the Language Model (LM) within the mapping network training. However, it is to be noted that fine-tuning the LM substantially escalates the number of trainable parameters (~156M for GPT2). As an alternative approach, the authors freeze the LM and replace the MLP mapping network with a transformer, effectively reducing the trainable parameters to ~43M.

To further optimize the model, we experiment with LoRA, a parameter efficient fine tuning technique. We apply LoRA to the baseline architecture (MLP mapper + GPT2) and our best-performing models. We also test it across all layers as well as a subset of layers of the LM.

### Configuration of Mappers


For all architectures utilising the MLP mapper we use the following hyperparameters:

| Parameter | Value                     |
|-------------------------|---------------------------|
| Hidden Layers           |     1                     |
| Hidden Layer Size    | 3840         |
| Activation              | Tanh                      |
| LM Prefix Length   | 10                |

The configuration of the transformer mapper:

| Parameter               | Value                     |
|-------------------------|---------------------------|
| Num Layers              | 8                         |
| Attention Heads         | 8                         |
| Embedding Dimension     | 768                       |
| Trainable Prefix Length   | 10                |
| LM Prefix Length   | 10                |


### Naming Conventions and Experimental Runs

We provide a table with naming conventions for different experimental runs to ensure clear understanding and easy reference.

(table)

## Methodology

We used CLIP-ViT/32, GPT-2, and FLANT5 models sourced from [Hugging Face's Transformers library](https://huggingface.co/docs/transformers/index). In order to maintain consistency with the original ClipCap approach, we have preserved all hyperparameters:

- Batch size: 40
- Learning rate: 2e-5
- Optimizer: AdamW
- Warm-up steps: 5000

In contrast to the original ClipCap implementation, which first extracted visual features with CLIP before proceeding to train different architectures, we adopted a simpler approach. Rather than dividing the procedure into two separate steps, we integrated CLIP into the training loop, allowing it to extract features at each training step. This method offers a better overview of the actual training time that such an implementation would take.

Our methodology involved running each model through 10 training epochs. To capture the model's optimal performance, we stored checkpoints throughout and identified the best one based on the lowest validation loss. This optimal checkpoint served as the basis for subsequent model evaluations, ensuring an accurate representation of the model's capabilities at its peak performance.

### Datasets

Choosing good datasets is a critical step for training and evaluating. The notion of "good dataset" in the context of visual-language tasks relies mainly on the diversity of context, topics and entities that the image and captions are covering. Following the original paper, we used the two datasets COCO and NOCAPS, both considered state-of-the-art datasets for image captioning modelling.

Similar to ClipCap work, we use COCO dataset to train models, and both, COCO and nocaps, to evaluate them.

The authors of the ClipCap paper also train their model on the large [Conceptual Caption (CoCa)](https://aclanthology.org/P18-1238/) dataset. However, due to the substantial computational resources and time required to process CoCa's extensive collection of over 3 million images, we opted not to use this dataset.


#### COCO

[COCO (Common Objects in Context)](https://arxiv.org/abs/1405.0312) is a large-scale dataset for image recognition, segmentation, and captioning. It contains over 200K images and 1.2M captions. We used the [Karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/) for our experiments, which is the same as used in ClipCap and [OSCAR](https://arxiv.org/abs/2004.06165). The Karpathy split divides the dataset into 113K training images, 5K validation images, and 5K test images.

We train our models on the training set, and perform evaluation on the test one.

#### NOCAPS

[NOCAPS (Novel Object CAPtioning dataset)](https://arxiv.org/abs/1812.08658) contains over 166K images and 1.5M captions. It is designed to measure the robustness and generalization of image captioning models to novel objects and concepts. It consists of three subsets: in-domain (similar categories than COCO), near-domain (similar categories than Open Images and COCO eventually), and out-domain (no shared categories or entities). Unlike direct annotation datasets such as COCO Captions, or filtered datasets such as Conceptual Captions, Nocaps has a larger and more diverse set of image classes and captions. Therefore, NOCAPS is more desirable than COCO or Conceptual Caption datasets, as it encourages the development of image captioning models that can learn visual concepts from other data sources, generalise better and therefore be considered more robust (Wang et al., 2022).

Same as in the ClipCap work, for the evaluation we use the validation set, which contains 9K images.

Specifically, we were more attentive to analyse results from the out-of-domain subset as it reflects the most challenging tasks. Models trained only on COCO data are likely to make [‘embarrassing errors’](https://arxiv.org/pdf/1612.00370.pdf) on this subset, reflecting the current performance of COCO trained models in the wild.

### Evaluation

We evaluated our models on both quantitative and qualitative metrics.

#### Generation / Inference

For the caption generation, the exact procedure from the original ClipCap paper is not clearly defined. To ensure consistency across our evaluations, we decided to implement a uniform approach by adopting a greedy search algorithm for all models. This strategy picks the most likely word at each step in the sequence, with the maximum length of the caption set at 67 tokens.

#### Quantitative Evaluation

##### Evaluation metrics

Image captioning is a notoriously difficult task to evaluate due to its inherent ambiguity (Cui et al., 2018). Human evaluation scores are reliable but expensive to obtain and not reproducible. Thus, current image captioning models are usually evaluated with automatic evaluation metrics. Similar to the Clipcap paper, we validate our model over the COCO dataset using the considered state-of-the-art metrics [CIDEr](https://arxiv.org/abs/1411.5726) and [SPICE](https://arxiv.org/abs/1607.08822). We decided to discard [BLEU](https://aclanthology.org/P02-1040/), ROUGE-L and [METEOR](https://aclanthology.org/W14-3348/) now considered out-dated. 

Most of the metrics in common use for caption evaluation are based on n-gram matching and measure the word overlap and semantic similarity between the generated captions and the reference captions from the datasets. The most known ones are BLEU, ROUGE and METEOR. But recently (Anderson et al., 2016) they have been outdated in their evaluation range capabilities. Indeed, more complex and robustness-measuring metrics have been developed and they are now considered state-of-the-art metrics (Anderson et al., 2016). First, previous metrics were primarily sensitive to n-gram overlap which made them sensitive to the size of the dataset. On the other hand, the novel metrics are size-independent and have been shown to have the strongest correlation with human judgments (Liu et al., 2017) and have been used in prior novel object captioning work (Anderson et al., 2018; Hendricks et al., 2016; Lu et al., 2018). In particular, To overcome the limitations of existing n-gram based automatic evaluation metrics, SPICE hypothesises that semantic propositional content is an important component of human caption evaluation and estimates caption quality by transforming both candidate and reference captions into a graph-based semantic representation called a scene graph, which make it more content-equivariant. The scene graph explicitly encodes the objects, attributes and relationships found in image captions, abstracting away most of the lexical and syntactic idiosyncrasies of natural language in the process.

CIDEr [cider paper] applies term frequency-inverse document frequency (tfidf) weights to n-grams in the candidate and reference sentences, which are then compared by summing their cosine similarity across n-grams. It is worth noting that CIDEr score is the only one that ranges from 0 to infinity. The score is calculated using the average cosine similarity between the candidate sentence and the reference sentences. The score can be greater than 1 if the candidate sentence is more similar to the reference sentences than the reference sentences are to each other. Being an F-score, SPICE [spice paper] is simple to understand, and easily interpretable as it is naturally bounded between 0 and 1. Unlike CIDEr, SPICE does not use cross-dataset statistics, such as corpus word frequencies, and is therefore equally applicable to both small and large datasets. 
In summary, while CIDEr focuses on consensus and overall relevance, SPICE centers on semantic propositions. The CIDEr metric assesses how well the machine-generated caption aligns with the consensus annotations of human captions for the same image. If the caption reflects the overall content and significance of the image, and is similar to the consensus captions, it receives a high CIDEr score. The SPICE metric, on the other hand, evaluates the precision and recall of semantic propositions in the machine-generated caption. It analyses how accurately the caption represents the semantic relationships within the image. If the caption correctly identifies the presence of people, a picnic, and a park, and expresses their relationships accurately, it will receive a high SPICE score. In summary, while CIDEr focuses on consensus and overall relevance, SPICE centers on semantic propositions.

We evaluate on COCO following the [OSCAR methodology](http://arxiv.org/abs/2004.06165) and we rewrote the script to adapt it to our pipeline, using the test set only and reformating the annotation file before tokenize the captions with ptb tokenizer to finally pass them as dictionaries to pycocoevalcap which run CIDEr and SPICE.
We did the same with NOCAPS, reformating the annotation file in three subsets in order to submit it to pycocoevalcap. Our results were not exactly the same locally and on the official evaluation server [Evalai](https://eval.ai/web/challenges/challenge-page/355) so we show only metrics results from the official evaluation server.

In order to rank our models we were interested in reporting their main characteristics  : their *total number of parameters*, their *total number of trainable parameters* as it’s a popular measure to indicate model feasibility, and the *estimated training time* (as it also includes the warm-up phase and the evaluation). Less trainable parameters is link to faster convergence time (even if it's not the only factor). Total number of parameters would influence the inference speed (all other architecture differences being equal).



#### Qualitative Evaluation

We conduct the qualitative evaluation by generating the captions of the five first images of the coco dataset and 3 images of the NOCAPS, one in domain, one near domain and one out of domain. We conduct human evaluation using [THumB](https://arxiv.org/pdf/2111.08940.pdf), a rubric-based protocol that assesses the quality of captions along two main dimensions: precision (how accurate and relevant the caption is) and recall (how much salient information the caption covers) and is designed to promote the human evaluation transparency for qualitative evaluation. For each model, we show the generation of the first 2 coco images, on of in-domain, 1 of near-domain, 1 of out-domain. First we define the precision of the caption out of 5 counting the number of false positive (hallucinations). Then we define Recall which measure how much of the salient information from the image is covered by the caption. For instance, an otter is a small animal, and thus small animal is precise. However, it is much less informative (and less natural) than saying an otter. Finally, we add a penalty based on the [fluency](https://arxiv.org/pdf/2111.08940.pdf) of the sentence from 0 to 1 if there is weird repetitions, misspellings or grammatical errors. There is also the Conciseness and the Inclusive Language to take into account but we did not target any problems.

## Results

| Images                      |                                                             |             |     |                                                       |             |     |                                                              |       |     |
|-----------------------------|-------------------------------------------------------------|-------------|-----|-------------------------------------------------------|-------------|-----|--------------------------------------------------------------|-------|-----|
| Clipcap_mlp                 | A boy standing in front of a wooden bench.                  | P4 R4       | 4   | A man riding on a elephant with a man on top.         | P2 R2 F-0.5 | 1.5 | A coffee and a bottle of soda on a table.                    | P4 R3 | 3.5 |
| Clipcap_mlp_ft              | A young boy standing next to a parked motorcycle.           | P3 R4       | 3.5 | A man riding on the back of an elephant.              | P2 R2       | 2   | A table topped with a cup of coffee and a box of ice cream.  | P2 R3 | 2.5 |
| Clipcap_mlp_proj_ft         | A little boy standing on a sidewalk holding a toothbrush.   | P2 R4       | 3   | A man riding on the back of an elephant.              | P2 R2       | 2   | A table topped with a bag of drinks and a bag of snacks.     | P4 R3 | 3.5 |
| Clipcap_trans               | A young boy is standing in a wooden bench.                  | P4 R4 F-0.5 | 3.5 | A man riding on top of an elephant with a man on top. | P2 R2 F-0.5 | 1.5 | A table with a bunch of drinks and a cup of coffee.          | P3 R3 | 3   |
| Flan_mlp_base_proj          | A little boy is standing on a sidewalk.                     | P4 R4       | 4   | An elephant with a man on it's back.                  | P3 R2       | 2.5 | A bunch of sodas and a mug of beer.                          | P3 R3 | 3   |
| Flan_mlp_base_proj_ft       | A young boy standing on a sidewalk holding a tennis racket. | P3 R4       | 3.5 | A man riding on the back of an elephant.              | P2 R2       | 2   | A table topped with a cup of coffee and a soda.              | P3 R3 | 3   |
| Flan_mlp_base_proj_lora_all | A little boy is standing in the street.                     | P5 R4       | 4.5 | A man riding an elephant on a dirt road.              | P3 R2       | 2.5 | A variety of different types of drinks are on a table.       | P5 R4 | 4.5 |
| Flan_mlp_large_proj         | A young child standing on a sidewalk with a hat.            | P3 R4       | 3.5 | A man is riding on top of an elephant.                | P2 R2       | 2   | A can of soda and a bottle of a cola.                        | P3 R3 | 3   |
| Flan_mlp_small_proj         | A little boy in a shirt and a shirt.                        | P5 R4 F-0.5 | 4   | A large elephant with a tusk on its back.             | P3 R2       | 2.5 | A group of various types of food and drinks.                 | P5 R3 | 4   |
| Flan_mlp_small_proj_ft      | A young boy is standing on the sidewalk.                    | P4 R4       | 4   | A man riding on the back of an elephant.              | P2 R2       | 2   | A bunch of drinks and a bottle of Coca Cola.                 | P3 R3 | 3   |
| Flant5_base_ft              | A young boy wearing a tie and a hat.                        | P2 R4       | 3   | A man riding an elephant on a dirt road.              | P2 R2       | 2   | A table with a cup of coffee, a drink and a bottle of water. | P2 R3 | 2.5 |


| Images                      | *In domain* ![](images/coco_1.png)            | *Near-domain*  ![](images/nocaps_2.png)          | *Out-domain* ![](images/nocaps_3.png)              |
|-----------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Clipcap_mlp                 | A boy standing in front of a wooden bench. <br/>[4, 4, 4, 0]                    | A man riding on a elephant with a man on top. <br/>[1.5, 2, 2, -0.5]         | A coffee and a bottle of soda on a table. <br/>[3.5, 4, 3, 0]                    |
| Clipcap_mlp_ft              | A young boy standing next to a parked motorcycle. <br/>[3.5, 3, 4,0 ]           | A man riding on the back of an elephant. <br/>[2, 2, 2, 0]                    | A table topped with a cup of coffee and a box of ice cream. <br/>[2.5, 2, 3, 0]  |
| Clipcap_mlp_proj_ft         | A little boy standing on a sidewalk holding a toothbrush. <br/>[3, 2, 4, 0]     | A man riding on the back of an elephant. <br/>[2, 2, 2, 0]                    | A table topped with a bag of drinks and a bag of snacks. <br/>[3.5, 4, 3, 0]     |
| Clipcap_trans               | A young boy is standing in a wooden bench. <br/>[3.5, 4, 4, -0.5]              | A man riding on top of an elephant with a man on top. <br/>[1.5, 2, 2, -0.5] | A table with a bunch of drinks and a cup of coffee. <br/>[3, 3, 3, 0]            |
| Flan_mlp_base_proj          | A little boy is standing on a sidewalk. <br/>[4, 4, 4, 0]                       | An elephant with a man on it's back. <br/>[**2.5**, 3, 2, 0]                      | A bunch of sodas and a mug of beer. <br/>[3, 3, 3, ]                            |
| Flan_mlp_base_proj_ft       | A young boy standing on a sidewalk holding a tennis racket. <br/>[3.5, 3, 4, 0] | A man riding on the back of an elephant. <br/>[2, 2, 2, 0]                    | A table topped with a cup of coffee and a soda. <br/>[3, 3, 3, 0]                |
| Flan_mlp_base_proj_lora_all | A little boy is standing in the street. <br/>[**4.5**, 5, 4, 0]                     | A man riding an elephant on a dirt road. <br/>[**2.5**, 3, 2, 0]                  | A variety of different types of drinks are on a table. <br/>[**4.5**, 5, 4, 0]       |
| Flan_mlp_large_proj         | A young child standing on a sidewalk with a hat. <br/>[3.5, 3, 4, 0]            | A man is riding on top of an elephant. <br/>[2, 2, 2, 0]                      | A can of soda and a bottle of a cola. <br/>[3, 3, 3, 0]                          |
| Flan_mlp_small_proj         | A little boy in a shirt and a shirt. <br/>[4, 5, 4, -0.5]                      | A large elephant with a tusk on its back. <br/>[**2.5**, 3, 2, 0]                 | A group of various types of food and drinks. <br/>[4, 5, 3, 0]                   |
| Flan_mlp_small_proj_ft      | A young boy is standing on the sidewalk. <br/>[4, 4, 4, 0]                      | A man riding on the back of an elephant. <br/>[2, 2, 2, 0]                    | A bunch of drinks and a bottle of Coca Cola. <br/>[3, 3, 3, 0]                   |
| Flant5_base_ft              | A young boy wearing a tie and a hat. <br/>[3, 2, 4, 0]                                     | A man riding an elephant on a dirt road. <br/>[2, 2, 2, 0]                    | A table with a cup of coffee, a drink and a bottle of water. <br/>[2.5, 2, 3, 0] |


### Baseline Runs

(table and/or plots)

Using CIDEr and SPICE scores on COCO dataset as our primary evaluation metrics, we observed results that didn't precisely match those reported in the original ClipCap paper. It's important to note here that the disparity might be due to the different methods of caption generation employed, given that the exact procedure was not explicitly stated in the original paper, as previously mentioned.

Nonetheless, a significant validation of our approach was that our training and validation loss matched those from the original ClipCap repository when using default parameters. This consistency suggests that our training procedure was robust, despite the discrepancies in caption generation outcomes.

As for the training time, there was a noticeable increase in our case compared to the original paper. This increase can be attributed to our decision to include CLIP in the forward pass of our model. Unlike the original work, where visual feature extraction was a separate step, we integrated this process within the training loop, as mentioned in an earlier section.

### Different Language Models

| Language Model | LM Type         | LM Size | Finetuning LM | Mapper                             | CIDEr       | SPICE       | Runtime(Hours) | Total Parameters(M) |
| -------------- | --------------- | ------- | ------------- | ---------------------------------- | ----------- | ----------- | -------------- | ------------------- |
| GPT2           | Decoder Only    | base    | Frozen        | Pooled embeddings with MLP         | 92.05680479 | 13.31473699 | 9.87           | 244                 |
| GPT2 (Baseline from ClipCap Paper)         | Decoder Only    | base    | Frozen        | Pooled embeddings with Transformer | 91.57263087 | 13.4573477  | 10.77          | 254                 |
| GPT2 (Baseline from ClipCap Paper)          | Decoder Only    | base    | Finetuned     | Pooled embeddings with MLP         | 101.5880106 | 14.15982716 | 12.91          | 244                 |
| FLAN-T5        | Decoder Only    | base    | Frozen        | Pooled embeddings with Transformer | 93.62617392 | 18.87172958 | 10.4           | 292                 |
| FLAN-T5        | Decoder Only    | small   | Frozen        | Pooled embeddings with Transformer | 93.19173182 | 18.20065787 | 6.44           | 165                 |
| FLAN-T5        | Encoder Decoder | base    | Finetuned     | Pooled embeddings with MLP         | 105.520692  | 19.4933659  | 13.16          | 367                 |
| FLAN-T5        | Encoder Decoder | small   | Finetuned     | Pooled embeddings with MLP         | 95.13775443 | 18.0817555  | 6.6            | 186                 |

### Utilisation of Unpooled Visual Representations

#### Ablation Study on Hidden Layer Size

Here we perform an ablation study investigating impact of the hidden layer size of the MLP on

### Application of LoRA


### T5 Weights

Additionally, we have performed a comparison of performance on selected FLAN-T5-based architecture with different weights - FLAN-T5 and original T5 ones.

(cider score chart)

(captions)

### Qualitative Results

| Column 1 | Column 2 | Column 3 |
| -------- | -------- | -------- |
| Text     | Text     | Text     |


## Discussion



## Further Work

There is considerable potential for further exploration and refinement in our modified ClipCap model. One potential avenue for future research involves experimenting with the multi-layer perceptron (MLP) for unpooled representations. Changing variables such as depth and activation functions could have a significant impact on performance and offer valuable insights into the optimal configuration for this element of the model.

In addition, we see value in examining the integration of global information alongside the unpooled CLIP representations. The hypothesis is that in the current approach, the processed visual tokens contain mostly information about local content of an image patch, which could be enhanced by providing a broader context. Integrating global information could potentially deliver a more comprehensive picture of the visual data, thus further improving captioning performance. However, this remains a hypothesis and will require rigorous testing and validation.


## Individual Contribution



## References

: X
: Agrawal, H., Desai, K., Wang, Y., Chen, X., Jain, R., Johnson, M., Batra, D., Parikh, D., Lee, S., & Anderson, P. (2019). nocaps : Novel object captioning at scale. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 8947‑8956. https://doi.org/10.1109/ICCV.2019.00904
: Anderson, P., Fernando, B., Johnson, M., & Gould, S. (2016). SPICE : Semantic Propositional Image Caption Evaluation (arXiv:1607.08822). arXiv. http://arxiv.org/abs/1607.08822
: Anderson, P., Gould, S., & Johnson, M. (2018). Partially-Supervised Image Captioning (arXiv:1806.06004). arXiv. http://arxiv.org/abs/1806.06004
Cui, Y., Yang, G., Veit, A., Huang, X., & Belongie, S. (2018). Learning to Evaluate Image Captioning (arXiv:1806.06422; Version 1). arXiv. http://arxiv.org/abs/1806.06422
: Hendricks, L. A., Venugopalan, S., Rohrbach, M., Mooney, R., Saenko, K., & Darrell, T. (2016). Deep Compositional Captioning : Describing Novel Object Categories without Paired Training Data (arXiv:1511.05284). arXiv. http://arxiv.org/abs/1511.05284
: Kasai, J., Sakaguchi, K., Dunagan, L., Morrison, J., Bras, R. L., Choi, Y., & Smith, N. A. (2022). Transparent Human Evaluation for Image Captioning (arXiv:2111.08940). arXiv. http://arxiv.org/abs/2111.08940
: Li, X., Yin, X., Li, C., Zhang, P., Hu, X., Zhang, L., Wang, L., Hu, H., Dong, L., Wei, F., Choi, Y., & Gao, J. (2020). Oscar : Object-Semantics Aligned Pre-training for Vision-Language Tasks (arXiv:2004.06165; Version 2). arXiv. http://arxiv.org/abs/2004.06165
: Liu, S., Zhu, Z., Ye, N., Guadarrama, S., & Murphy, K. (2017). Improved Image Captioning via Policy Gradient optimization of SPIDEr. 2017 IEEE International Conference on Computer Vision (ICCV), 873‑881. https://doi.org/10.1109/ICCV.2017.100
: Lu, J., Yang, J., Batra, D., & Parikh, D. (2018). Neural Baby Talk (arXiv:1803.09845). arXiv. http://arxiv.org/abs/1803.09845
: Mokady, R., Hertz, A., & Bermano, A. H. (2021). ClipCap : CLIP Prefix for Image Captioning (arXiv:2111.09734). arXiv. http://arxiv.org/abs/2111.09734
: Vedantam, R., Zitnick, C. L., & Parikh, D. (2015). CIDEr : Consensus-based Image Description Evaluation (arXiv:1411.5726). arXiv. http://arxiv.org/abs/1411.5726
: Wang, X., Wang, H., & Yang, D. (2022a). Measure and Improve Robustness in NLP Models : A Survey. Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 4569‑4586. https://doi.org/10.18653/v1/2022.naacl-main.339
: Wang, X., Wang, H., & Yang, D. (2022b). Measure and Improve Robustness in NLP Models : A Survey. Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 4569‑4586. https://doi.org/10.18653/v1/2022.naacl-main.339
