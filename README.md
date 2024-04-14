# Binding Text, Images, Graphs, and Audio for Music Representation Learning
This repo is a work in progress.
This repo contains the [code](https://github.com/a-tabaza/binding_music/blob/main/fairouz/notebooks/model.ipynb) for inference, and [evaluation](https://github.com/a-tabaza/binding_music/blob/main/fairouz/checkpoints/code/load_checkpoint.ipynb) for the paper ``Binding Text, Images, Graphs, and Audio for Music Representation Learning`` 

The current state of this repo is not ideal, to help you navigate around checkpoints and inference, please refer to the following [sheet](https://docs.google.com/spreadsheets/d/11v6GrVe-0SJwl2Xqv_F5k20S1s6H_6B1uQ6oYKZL2Z0/edit?usp=sharing) temporarily while we prepare this repo.
The code for embedding Text and Images is availabe in the scripts folder. For Audio Embeddings, code is available [here](https://github.com/a-tabaza/audio_embeddings), for Graph Embeddings, code is available [here](https://github.com/AbdelRahmanYaghi/FairouzConf)

N.B. _Fairouz_ refers to the codename given to the model we envisioned, this is an _iteration_, hopefully of many, it covers part of our vision, but nowhere near the full scope of what we aim to do with _Fairouz_

## Abstract
In the field of Information Retrieval and Natural Language Processing, text embeddings play a significant role in tasks such as classification, clustering, and topic modeling. However, extending these embeddings to abstract concepts such as music, which involves multiple modalities, presents a unique challenge. Our work addresses this challenge by integrating rich multi-modal data into a unified joint embedding space. This space includes textual, visual, acoustic, and graph-based modality features. By doing so, we mirror cognitive processes associated with music interaction and overcome the disjoint nature of individual modalities. The resulting joint low-dimensional vector space facilitates retrieval, clustering, embedding space arithmetic, and cross-modal retrieval tasks. Importantly, our approach carries implications for music information retrieval and recommendation systems. Furthermore, we propose a novel multi-modal model that integrates various data types—text, images, graphs, and audio—for music representation learning. Our model aims to capture the complex relationships between different modalities, enhancing the overall understanding of music. By combining textual descriptions, visual imagery, graph-based structures, and audio signals, we create a comprehensive representation that can be leveraged for a wide range of music-related tasks. Notably, our model demonstrates promising results in music classification, recommendation systems.

### Nomic Maps
#### Text Embedding Maps
- [BAAI/bge-large-en-v1.5](https://atlas.nomic.ai/data/omaralquishawi25/model-bge-1/map-)
- [intfloat/e5-large-v2](https://atlas.nomic.ai/data/omaralquishawi25/model-e5-1/map)
- [jinaai/jina-embeddings-v2-base-en](https://atlas.nomic.ai/data/omaralquishawi25/model-jina-1/map)
- [mixedbread-ai/mxbai-embed-large-v1](https://atlas.nomic.ai/data/omaralquishawi25/model-mxbai/map)

#### Image Embedding Maps
- [dinov2_vitb14](https://atlas.nomic.ai/data/omaralquishawi25/model-dino-1/map)
- [CLIP-ViT-B-32-laion2B](https://atlas.nomic.ai/data/omaralquishawi25/model-openclip-1/map)

#### Graph Embedding Maps
- [Role2Vec](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-role2vec/map)
- [Node2Vec](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-node2vec/map)
- [RandNE](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-randne/map)
- [GraphWave](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-graphwave/map)
- [DeepWalk](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-deepwalk/map)

#### Audio Embedding Maps
- [Vggish](https://atlas.nomic.ai/data/omaralquishawi25/all-music-embeddings-march-23rd---mean/map)
- [L3](https://atlas.nomic.ai/data/omaralquishawi25/model-openl3/map)

#### Multimodal Embedding Maps
- [51k Data Pairs](https://atlas.nomic.ai/data/tyqnology/fairouz-vggish-randne-openclip-mxbai-200-epochs-contracted-51k-datapoints-euclidian/map)
- [6k Data Pairs](https://atlas.nomic.ai/data/tyqnology/fairouz-vggish-randne-openclip-mxbai-200-epochs-contracted-dropout-euclidian/map) 
