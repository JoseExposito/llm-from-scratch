from enum import Enum
from transformers import PreTrainedTokenizerFast
from typing import Literal


class NormalizationStrategy(Enum):
    """Estrategias de normalización disponibles para el modelo"""

    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"


class PositionalEmbeddingStrategy(Enum):
    """Estrategias de embedding posicional disponibles para el modelo"""

    ABSOLUTE = "absolute"
    ROPE = "rope"


class Config:
    """Clase que almacena la configuración utilizada por el modelo"""

    def __init__(
        self,
        name: str,
        tokenizer: object,
        vocabulary_size: int,
        context_length: int,
        embedding_dim: int,
        n_heads: int,
        n_transformer_blocks: int,
        dropout_rate: float,
        query_key_value_bias: bool,
        normalization_strategy: NormalizationStrategy,
        positional_embedding_strategy: PositionalEmbeddingStrategy,
    ) -> None:
        """
        Args:
            name: Nombre de la configuración
            tokenizer: Tokenizador utilizado por el modelo. Debe implementar los
                métodos encode() y decode().
            vocabulary_size: Tamaño del vocabulario, número de embedding IDs
            context_length: Tamaño del contexto. Es el número de embedding
                tokens a los que el modelo puede prestar atención. También es
                el número de embedding tokens a los que el modelo puede asignar
                información posicional.
            embedding_dim: Número de dimensiones de cada embedding.
            n_heads: Número de "cabezas" (heads) del mecanismo de atención.
            n_transformer_blocks: Número de bloques de transformers utilizados
                por el modelo.
            dropout_rate: Porcentaje (de 0 a 1) de embedding tokens a enmascarar
                durante el entrenamiento. Ayuda a reducir el overfitting.
            query_key_value_bias: Si se suma o no un bias a las neuronas de las
                capas de query, key y value del mecanismo de atención.
            normalization_strategy: Estrategia de normalización utilizada en los
                bloques transformer y en la normalización final del modelo.
            positional_embedding_strategy: Estrategia de embedding posicional
                utilizada por el modelo para codificar la posición de los tokens
                en la secuencia.
        """
        self.name = name
        self.tokenizer = tokenizer
        self.vocabulary_size = vocabulary_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_transformer_blocks = n_transformer_blocks
        self.dropout_rate = dropout_rate
        self.query_key_value_bias = query_key_value_bias
        self.normalization_strategy = normalization_strategy
        self.positional_embedding_strategy = positional_embedding_strategy


class ConfigFactory:
    """Factoría para crear las distintas configuraciones disponibles"""

    @staticmethod
    def _load_tinystories_tokenizer() -> tuple[PreTrainedTokenizerFast, int]:
        # https://huggingface.co/vuiseng9/bpe-10.0k-tinystories
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "vuiseng9/bpe-10.0k-tinystories"
        )
        return tokenizer, tokenizer.vocab_size

    @staticmethod
    def create_config(
        type: Literal["base-model-10M", "rms-norm-10M", "rope-10M"],
    ) -> Config:
        if type == "base-model-10M":
            tokenizer, vocabulary_size = ConfigFactory._load_tinystories_tokenizer()

            return Config(
                name=type,
                tokenizer=tokenizer,
                vocabulary_size=vocabulary_size,
                context_length=512,
                embedding_dim=256,
                n_heads=8,
                n_transformer_blocks=6,
                dropout_rate=0.1,
                query_key_value_bias=False,
                normalization_strategy=NormalizationStrategy.LAYER_NORM,
                positional_embedding_strategy=PositionalEmbeddingStrategy.ABSOLUTE,
            )
        elif type == "rms-norm-10M":
            tokenizer, vocabulary_size = ConfigFactory._load_tinystories_tokenizer()

            return Config(
                name=type,
                tokenizer=tokenizer,
                vocabulary_size=vocabulary_size,
                context_length=512,
                embedding_dim=256,
                n_heads=8,
                n_transformer_blocks=6,
                dropout_rate=0.1,
                query_key_value_bias=False,
                normalization_strategy=NormalizationStrategy.RMS_NORM,
                positional_embedding_strategy=PositionalEmbeddingStrategy.ABSOLUTE,
            )
        elif type == "rope-10M":
            tokenizer, vocabulary_size = ConfigFactory._load_tinystories_tokenizer()

            return Config(
                name=type,
                tokenizer=tokenizer,
                vocabulary_size=vocabulary_size,
                context_length=512,
                embedding_dim=256,
                n_heads=8,
                n_transformer_blocks=6,
                dropout_rate=0.1,
                query_key_value_bias=False,
                normalization_strategy=NormalizationStrategy.LAYER_NORM,
                positional_embedding_strategy=PositionalEmbeddingStrategy.ROPE,
            )
        raise NotImplementedError(f"La configuración {type} no está soportada")
