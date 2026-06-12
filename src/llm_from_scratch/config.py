import tiktoken
from typing import Literal

class Config:
    """Clase que almacena la configuración utilizada por el modelo"""
    def __init__(
                self,
                vocab_size: int,
                context_length: int,
                embedding_dimensions: int,
                n_heads: int,
                n_transformer_blocks: int,
                dropout_rate: float,
                query_key_value_bias: bool,
            ) -> None:
        """
        Args:
            vocab_size: Tamaño del vocabulario, número de embedding IDs
            context_length: Tamaño del contexto. Es el número de embedding
                tokens a los que el modelo puede prestar atención. También es
                el número de embedding tokens a los que el modelo puede asignar
                información posicional.
            embedding_dimensions: Número de dimensiones de cada embedding.
            n_heads: Número de "cabezas" (heads) del mecanismo de atención.
            n_transformer_blocks: Número de bloques de transformers utilizados
                por el modelo.
            dropout_rate: Porcentaje (de 0 a 1) de embedding tokens a enmascarar
                durante el entrenamiento. Ayuda a reducir el overfitting.
            query_key_value_bias: Si se suma o no un bias a las neuronas de las
                capas de query, key y value del mecanismo de atención.
        """
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dimensions = embedding_dimensions
        self.n_heads = n_heads
        self.n_transformer_blocks = n_transformer_blocks
        self.dropout_rate = dropout_rate
        self.query_key_value_bias = query_key_value_bias


class ConfigFactory:
    """Factoría para crear las distintas configuraciones disponibles"""

    @staticmethod
    def create_config(type: Literal["gpt-2"]) -> Config:
        vocab_size = tiktoken.get_encoding("gpt2").n_vocab

        if type == "gpt-2":
            return Config(
                vocab_size = vocab_size,
                context_length = 1024,
                embedding_dimensions = 768,
                n_heads = 12,
                n_transformer_blocks = 12,
                dropout_rate = 0.1,
                query_key_value_bias = False,
            )

        raise NotImplementedError(f"La configuración {type} no está soportada")


