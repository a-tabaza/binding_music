# Binding Text, Images, Graphs, and Audio for Music Representation Learning
This repo contains the code for training, inference, and evaluation for the paper [``Binding Text, Images, Graphs, and Audio for Music Representation Learning``](https://doi.org/10.1145/3660853.3660886)

To help you navigate around checkpoints and inference, please refer to the following [sheet](https://docs.google.com/spreadsheets/d/11v6GrVe-0SJwl2Xqv_F5k20S1s6H_6B1uQ6oYKZL2Z0/edit?usp=sharing).

The code for embedding Text and Images is availabe in the scripts folder. For Audio Embeddings, code is available [here](https://github.com/a-tabaza/audio_embeddings), for Graph Embeddings, code is available [here](https://github.com/AbdelRahmanYaghi/FairouzConf)

We also provide a simple demo that showcases the model's predictions with explanations. The demo is available [here](https://fairouz.streamlit.app/) and the code is available [here](https://github.com/a-tabaza/fairouz_demo), you can run the demo locally as well with instructions provided in the repo.

## Repo Structure:
- `scripts/` contains the code for setting up embedding APIs for Text and Images, and the LLM API, as well as code for downloading model weights.
- `data/` contains JSON files with tracks and their metadata, as well as our positives and negatives for training
- `modelling/` contains the code for the multimodal model, to use the modules, refer to the sheet mentioned above, each architecture has a different script that defines its architecture
- `embeddings/` contains JSON files with embeddings for each modality, as well as the multimodal embeddings
- `checkpoints/` contains the model weights for the multimodal model
- `notebooks/` contains notebooks for evaluation and inference

## Abstract
In the field of Information Retrieval and Natural Language Processing, text embeddings play a significant role in tasks such as classification, clustering, and topic modeling. However, extending these embeddings to abstract concepts such as music, which involves multiple modalities, presents a unique challenge. Our work addresses this challenge by integrating rich multi-modal data into a unified joint embedding space. This space includes textual, visual, acoustic, and graph-based modality features. By doing so, we mirror cognitive processes associated with music interaction and overcome the disjoint nature of individual modalities. The resulting joint low-dimensional vector space facilitates retrieval, clustering, embedding space arithmetic, and cross-modal retrieval tasks. Importantly, our approach carries implications for music information retrieval and recommendation systems. Furthermore, we propose a novel multi-modal model that integrates various data types—text, images, graphs, and audio—for music representation learning. Our model aims to capture the complex relationships between different modalities, enhancing the overall understanding of music. By combining textual descriptions, visual imagery, graph-based structures, and audio signals, we create a comprehensive representation that can be leveraged for a wide range of music-related tasks. Notably, our model demonstrates promising results in music classification and recommendation systems.

## Nomic Maps
### Text Embedding Maps
- [BAAI/bge-large-en-v1.5](https://atlas.nomic.ai/data/omaralquishawi25/model-bge-1/map-)
- [intfloat/e5-large-v2](https://atlas.nomic.ai/data/omaralquishawi25/model-e5-1/map)
- [jinaai/jina-embeddings-v2-base-en](https://atlas.nomic.ai/data/omaralquishawi25/model-jina-1/map)
- [mixedbread-ai/mxbai-embed-large-v1](https://atlas.nomic.ai/data/omaralquishawi25/model-mxbai/map)

### Image Embedding Maps
- [dinov2_vitb14](https://atlas.nomic.ai/data/omaralquishawi25/model-dino-1/map)
- [CLIP-ViT-B-32-laion2B](https://atlas.nomic.ai/data/omaralquishawi25/model-openclip-1/map)

### Graph Embedding Maps
- [Role2Vec](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-role2vec/map)
- [Node2Vec](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-node2vec/map)
- [RandNE](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-randne/map)
- [GraphWave](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-graphwave/map)
- [DeepWalk](https://atlas.nomic.ai/data/abd20200355/graph-embedding-map-for-model-deepwalk/map)

### Audio Embedding Maps
- [Vggish](https://atlas.nomic.ai/data/omaralquishawi25/all-music-embeddings-march-23rd---mean/map)
- [L3](https://atlas.nomic.ai/data/omaralquishawi25/model-openl3/map)

### Multimodal Embedding Maps
- [51k Data Pairs](https://atlas.nomic.ai/data/tyqnology/fairouz-vggish-randne-openclip-mxbai-200-epochs-contracted-51k-datapoints-euclidian/map)
- [6k Data Pairs](https://atlas.nomic.ai/data/tyqnology/fairouz-vggish-randne-openclip-mxbai-200-epochs-contracted-dropout-euclidian/map) 

## Citation
@inproceedings{10.1145/3660853.3660886,
author = {Tabaza, Abdulrahman and Quishawi, Omar and Yaghi, Abdelrahman and Qawasmeh, Omar},
title = {Binding Text, Images, Graphs, and Audio for Music Representation Learning},
year = {2024},
isbn = {9798400716928},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3660853.3660886},
doi = {10.1145/3660853.3660886},
abstract = {Abstract In the field of Information Retrieval and Natural Language Processing, text embeddings play a significant role in tasks such as classification, clustering, and topic modeling. However, extending these embeddings to abstract concepts such as music, which involves multiple modalities, presents a unique challenge. Our work addresses this challenge by integrating rich multi-modal data into a unified joint embedding space. This space includes: (1) textual, (2) visual, (3) acoustic, and (4) graph-based modality features. By doing so, we mirror cognitive processes associated with music interaction and overcome the disjoint nature of individual modalities. The resulting joint low-dimensional vector space facilitates retrieval, clustering, embedding space arithmetic, and cross-modal retrieval tasks. Importantly, our approach carries implications for music information retrieval and recommendation systems. Furthermore, we propose a novel multi-modal model that integrates various data types—text, images, graphs, and audio—for music representation learning. Our model aims to capture the complex relationships between different modalities, enhancing the overall understanding of music. By combining textual descriptions, visual imagery, graph-based structures, and audio signals, we create a comprehensive representation that can be leveraged for a wide range of music-related tasks. Notably, our model demonstrates promising results in music classification, and recommendation systems. Code Availability: The source code for the multi-modal music representation model described in this paper is available on GitHub. Access and further details can be found at the following repository link: //github.com/a-tabaza/binding_music/},
booktitle = {Proceedings of the Cognitive Models and Artificial Intelligence Conference},
pages = {139–146},
numpages = {8},
location = {undefinedstanbul, Turkiye},
series = {AICCONF '24}
}
