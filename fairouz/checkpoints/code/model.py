import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class Encoder(L.LightningModule):
    def __init__(
        self,
        audio_size,
        image_size,
        text_size,
        graph_size,
        expansion_factor,
        contraction_factor,
        embedding_size,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_size, audio_size // contraction_factor), nn.ReLU()
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(image_size, image_size // contraction_factor), nn.ReLU()
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_size, text_size // contraction_factor), nn.ReLU()
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_size, graph_size // contraction_factor), nn.ReLU()
        )
        self.cat_size = (
            audio_size // contraction_factor
            + image_size // contraction_factor
            + text_size // contraction_factor
            + graph_size // contraction_factor
        )
        self.combined_encoder = nn.Sequential(
            nn.Linear(self.cat_size, self.cat_size * expansion_factor),
            nn.ReLU(),
            nn.Linear(self.cat_size * expansion_factor, embedding_size),
        )
        self.distance_metric = lambda x, y: F.pairwise_distance(x, y, p=2)
        self.margin = 0.5
        self.save_hyperparameters()

    def forward(self, anchor, query, labels):
        anchor_audio, anchor_image, anchor_text, anchor_graph = anchor
        query_audio, query_image, query_text, query_graph = query
        anchor_embedding = self.combined_encoder(
            torch.cat(
                (
                    self.audio_encoder(anchor_audio),
                    self.image_encoder(anchor_image),
                    self.text_encoder(anchor_text),
                    self.graph_encoder(anchor_graph),
                ),
                dim=1,
            )
        )
        query_embedding = self.combined_encoder(
            torch.cat(
                (
                    self.audio_encoder(query_audio),
                    self.image_encoder(query_image),
                    self.text_encoder(query_text),
                    self.graph_encoder(query_graph),
                ),
                dim=1,
            )
        )
        return anchor_embedding, query_embedding, labels

    def training_step(self, batch, batch_idx):
        anchor, query, labels = batch
        anchor_embedding, query_embedding, labels = self(anchor, query, labels)
        distances = self.distance_metric(anchor_embedding, query_embedding)
        losses = 0.5 * labels[0].float() * distances.pow(2) + (
            1 - labels[0]
        ).float() * F.relu(self.margin - distances).pow(2)
        loss = losses.mean()
        self.log("train_loss", loss)
        return loss

    def predict_step(self, audio, image, text, graph):
        audio = self.audio_encoder(audio)
        image = self.image_encoder(image)
        text = self.text_encoder(text)
        graph = self.graph_encoder(graph)
        embedding = self.combined_encoder(torch.cat((audio, image, text, graph), dim=1))
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
