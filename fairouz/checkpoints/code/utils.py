def transform_dict(list_of_dict):
    transformed_embedding = {}
    for dictionary in list_of_dict:
        transformed_embedding[dictionary["id"]] = dictionary["embedding"]

    return transformed_embedding


def get_modality_embeddings(
    track_id,
    audio_embeddings_dict,
    image_embeddings_dict,
    text_embeddings_dict,
    graph_embeddings_dict,
):
    audio_embedding = audio_embeddings_dict[track_id]
    graph_embedding = graph_embeddings_dict[track_id]
    image_embedding = image_embeddings_dict[track_id]
    text_embedding = text_embeddings_dict[track_id]
    return audio_embedding, graph_embedding, image_embedding, text_embedding
