"""Internal model, preprocessing, and memory-planning helpers for HECTOR inference."""

from __future__ import annotations

import csv
import functools
import logging
import math
import numpy as np
import os
from pathlib import Path
import random
import tempfile
import pandas as pd
from . import _HECTOR_USE_MLX as _USE_MLX
if not _USE_MLX:
    import tensorflow as tf
    from tensorflow.keras import layers
else:
    tf = None
    layers = None
import h5py
import re
import traceback
import warnings, random
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Literal
from dataclasses import dataclass, field
from importlib import resources
import anndata
from scipy import stats as scipy_stats
from scipy.sparse import issparse
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

logger = logging.getLogger(__name__)


class _ScoreMatrixBuffer:
    """CPU-backed score buffer that spills to a temporary memmap when needed."""

    def __init__(
        self,
        shape: Tuple[int, int],
        dtype: np.dtype = np.float32,
        *,
        prefer_memmap: bool = False,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._path: Optional[str] = None
        self._closed = False

        required_bytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        use_memmap = bool(prefer_memmap)

        if not use_memmap:
            try:
                import psutil

                available_memory = int(psutil.virtual_memory().available)
            except (ImportError, AttributeError):
                available_memory = 2 * 1024 ** 3

            in_memory_limit = min(int(available_memory * 0.25), 1024 ** 3)
            in_memory_limit = max(in_memory_limit, 256 * 1024 ** 2)
            use_memmap = required_bytes > in_memory_limit

        if use_memmap:
            with tempfile.NamedTemporaryFile(
                prefix="hector_scores_",
                suffix=".mmap",
                delete=False,
            ) as handle:
                self._path = handle.name
            self._array = np.memmap(
                self._path,
                mode="w+",
                dtype=self.dtype,
                shape=self.shape,
            )
        else:
            self._array = np.empty(self.shape, dtype=self.dtype)

    @property
    def array(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Score buffer has already been closed.")
        return self._array

    @property
    def is_memmap(self) -> bool:
        return isinstance(self._array, np.memmap)

    def flush(self) -> None:
        if self._closed:
            return
        if isinstance(self._array, np.memmap):
            self._array.flush()

    def close(self) -> None:
        if self._closed:
            return

        try:
            if isinstance(self._array, np.memmap):
                self._array.flush()
                mmap_obj = getattr(self._array, "_mmap", None)
                if mmap_obj is not None:
                    mmap_obj.close()
        finally:
            self._closed = True
            if self._path is not None and os.path.exists(self._path):
                try:
                    os.remove(self._path)
                except OSError:
                    pass


def _get_cpu_class_chunk_size(
    n_rows: int,
    n_classes: int,
    *,
    working_buffers: int = 3,
) -> int:
    """Estimate a safe class chunk size for CPU-side score processing."""

    try:
        import psutil

        available_memory = int(psutil.virtual_memory().available)
    except (ImportError, AttributeError):
        available_memory = 2 * 1024 ** 3

    if n_rows <= 0:
        return max(1, int(n_classes))

    bytes_per_class = int(n_rows) * 4 * max(1, working_buffers)
    chunk_size = int(available_memory) // max(1, bytes_per_class)
    chunk_size = max(1, min(chunk_size, int(n_classes)))
    return chunk_size


def stable_environments():
    if tf is not None:
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.experimental.enable_op_determinism()
# Seed the available numerical backends.
    seed=41
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


# ============================================================================
# Inference model layers
# ============================================================================

class ModelConfig:
    """Fallback config for the in-file GAT/HPL layers used during inference."""
    GAT_NUM_HEADS: int = 4
    GAT_NUM_LAYERS: int = 1
    GAT_DROPOUT: float = 0.1
    GAT_ATTENTION_DROPOUT: float = 0.1
    GAT_ACTIVATION: str = "relu"
    GAT_TRAINABLE: bool = True
    USE_LEVEL_EMBEDDINGS: bool = True
    LEVEL_EMBEDDING_DIM: int = 32
    HPL_LEVEL_EMBEDDING_DIM: int = 32

Config = ModelConfig()

if tf is not None:
    class HPLHead(tf.keras.layers.Layer):
        """
        Hierarchical proxy-based classification head.

        Instead of independent per-class weights, it learns weights per node
        in the ontology graph (node_parts). A class prototype is built
        dynamically as the sum of its ancestor vectors:
            W_class = AncestryMatrix @ node_parts = Sum(V_ancestors)

        Related cell types share ancestor vectors; for example, W_Naive and
        W_Memory can share V_Immune + V_TCell + V_CD4.

        Args:
            num_total_nodes: Number of ontology nodes
            embedding_dim: Cell and prototype embedding dimension
            ancestry_matrix: Binary matrix [num_seen_classes, num_total_nodes]
                            Maps training classes to their ancestor nodes
            scale: Scaling factor for logits (default: 20.0)

        Input:
            embeddings: Cell embeddings (mu) [batch_size, embedding_dim]

        Output:
            logits: Classification logits [batch_size, num_seen_classes]

        """

        def __init__(self, 
                     num_total_nodes: int,
                     embedding_dim: int,
                     ancestry_matrix: tf.Tensor,
                     scale: float = 20.0,
                     node_levels: np.ndarray = None,
                     **kwargs):
            """
            Initialize the hierarchical proxy-based classification head.

            Args:
                num_total_nodes: Number of ontology nodes
                embedding_dim: Cell and prototype embedding dimension
                ancestry_matrix: Binary matrix [num_seen_classes, num_total_nodes]
                                Maps training classes to their ancestor nodes
                scale: Initial scaling factor for logits (default: 20.0)
                node_levels: Optional integer array of BFS depth per node [num_total_nodes].
                            Used for learnable level embeddings when USE_LEVEL_EMBEDDINGS is True.
            """
            super().__init__(**kwargs)
            self.num_total_nodes = num_total_nodes
            self.embedding_dim = embedding_dim

            # Keep the checkpoint scale as a non-trainable weight.
            self.scale = self.add_weight(
                name='hpl_scale',
                shape=(),
                initializer=tf.constant_initializer(scale),
                trainable=False,
                dtype=tf.float32
            )

            # Store ancestry matrix as constant (not trainable)
            self.ancestry_matrix = tf.cast(ancestry_matrix, dtype=tf.float32)

            # Compute number of seen classes from ancestry matrix shape
            self.num_seen_classes = tf.shape(ancestry_matrix)[0].numpy()

            # Keep the angular margin as a non-trainable checkpoint weight.
            self.margin = self.add_weight(
                name='hpl_margin',
                shape=(),
                initializer=tf.constant_initializer(0.0),
                trainable=False,
                dtype=tf.float32
            )

            # Level embeddings for HPL prototype construction
            if Config.USE_LEVEL_EMBEDDINGS and node_levels is not None:
                if node_levels.shape[0] != num_total_nodes:
                    raise ValueError(
                        f"node_levels shape mismatch: got {node_levels.shape[0]} "
                        f"but expected {num_total_nodes} (num_total_nodes)"
                    )
                self.node_levels_np = node_levels.astype(np.int32).copy()
                self.max_level = int(node_levels.max())
                self.use_level_embeddings = True
            else:
                self.use_level_embeddings = False
                if Config.USE_LEVEL_EMBEDDINGS and node_levels is None:
                    print("[WARN] USE_LEVEL_EMBEDDINGS is True but node_levels not provided to HPLHead. "
                          "Falling back to node_parts-only prototype construction.")



        def build(self, input_shape):
            """Build the layer - creates trainable node_parts vectors and optional level embedding layers."""
            # Learn one vector for every ontology node, including unseen classes.
            self.node_parts = self.add_weight(
                name='node_parts',
                shape=(self.num_total_nodes, self.embedding_dim),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True,
                dtype=tf.float32
            )

            if self.use_level_embeddings:
                # Compact learnable embedding layer
                self.level_embedding_layer = tf.keras.layers.Embedding(
                    input_dim=self.max_level + 1,
                    output_dim=Config.HPL_LEVEL_EMBEDDING_DIM,
                    name="hpl_level_embedding"
                )
                # Explicitly build so weights appear in get_weights() immediately
                self.level_embedding_layer.build(input_shape=())

            # Project level features to node_parts width without an additive bias.
                self.level_projection = tf.keras.layers.Dense(
                    self.embedding_dim,
                    use_bias=False,
                    name="hpl_level_projection"
                )
                # Explicitly build so weights appear in get_weights() immediately
                self.level_projection.build(input_shape=(None, Config.HPL_LEVEL_EMBEDDING_DIM))

                # Non-trainable level tensor (same pattern as scale/margin add_weight)
                self.node_level_ids = self.add_weight(
                    name="node_level_ids",
                    shape=self.node_levels_np.shape,
                    dtype=tf.int32,
                    initializer=tf.constant_initializer(self.node_levels_np),
                    trainable=False
                )

            super().build(input_shape)

        def call(self, inputs, labels=None, training=None, class_embeddings=None):
            """
            Forward pass with optional ArcFace margin.

            Args:
                inputs: Cell embeddings (mu) [batch_size, embedding_dim]
                labels: Optional one-hot labels [batch_size, num_seen_classes]
                        or indices [batch_size]. None skips the angular margin.
                training: Whether in training mode.
                class_embeddings: Optional [num_seen_classes, embedding_dim] tensor.
                                When provided, external embeddings (e.g., GAT output)
                                replace ancestry-sum prototypes; node_parts receives
                                no prototype-construction gradients.
            """
            embeddings = inputs

            # 1. Construct class prototypes (proxies) for SEEN classes
            if class_embeddings is not None:
                # Use external prototypes when supplied.
                proxies = class_embeddings
            else:
                # Sum learned node vectors over each class's ancestors.
                # Include level features when level embeddings are enabled.
                # Shape: [num_seen_classes, num_total_nodes] @ [num_total_nodes, embedding_dim]
                # Each class prototype is the sum of its ancestor node vectors.
                augmented_parts = self.node_parts
                if self.use_level_embeddings:
                    level_emb = self.level_embedding_layer(tf.cast(self.node_level_ids, tf.int32))  # [N, level_dim]
                    level_proj = self.level_projection(level_emb)  # [N, embedding_dim]
                    augmented_parts = self.node_parts + level_proj
                proxies = tf.matmul(self.ancestry_matrix, augmented_parts)

            # 2. Normalize Embeddings and Proxies (Cosine Similarity)
            embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1, epsilon=1e-8)
            proxies_norm = tf.nn.l2_normalize(proxies, axis=1, epsilon=1e-8)

            # 3. Compute Cosine Similarity Logits
            # [batch, embedding_dim] @ [embedding_dim, num_seen_classes]
            cos_theta = tf.matmul(embeddings_norm, proxies_norm, transpose_b=True)

            # Early return if labels are not provided (cannot apply margin)
            if labels is None:
                return cos_theta * tf.cast(self.scale, cos_theta.dtype)

            # 4. Apply Angular Margin (ArcFace) - ONLY if training and margin > 0

            # Ensure labels are one-hot
            # If labels are indices (rank 1), convert to one-hot
            local_labels = labels
            if len(local_labels.shape) == 1 or local_labels.shape[-1] == 1:
                 # Use the class count from the ancestry matrix.
                 local_labels = tf.one_hot(tf.cast(local_labels, tf.int32), self.num_seen_classes)

            # Calculate target_logit = cos(theta + m)
            # Clamp away from the endpoints before taking acos.
            cos_theta_clamped = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
            theta = tf.math.acos(cos_theta_clamped)
            target_logit = tf.math.cos(theta + self.margin)

            # Apply the margin ONLY to the true class
            # logits = (1 - one_hot) * cos_theta + one_hot * target_logit
            logits_with_margin = cos_theta + (target_logit - cos_theta) * local_labels

            # Rescale
            logits_with_margin_scaled = logits_with_margin * self.scale
            logits_no_margin = cos_theta * self.scale

            # Check if we should apply margin: training=True AND margin > 0
            # Accept training as either a Python bool or a tensor.
            if training is None:
                training = False

            # Convert training to tensor if it's a python bool
            if isinstance(training, bool):
                training_tensor = tf.constant(training)
            else:
                training_tensor = training

            # Condition: training AND margin > 0
            condition = tf.logical_and(
                training_tensor, 
                tf.greater(self.margin, 0.0)
            )

            # Use tf.where instead of tf.cond (graph-compatible)
            return tf.where(condition, logits_with_margin_scaled, logits_no_margin)

    class GraphAttention(layers.Layer):
        """
        GATv2 Layer: Dynamic Graph Attention.

        GATv2 (Brody et al., 2021) fixes GATv1's limited expressiveness by swapping operations:
        - GATv1: LeakyReLU(a_src^T · W·h_i + a_dst^T · W·h_j)  [static attention]
        - GATv2: a^T · LeakyReLU(W·h_i + W·h_j)                [dynamic attention]

        Applying the nonlinearity before the attention projection allows
        attention to depend dynamically on the query.

        Includes structural bias from ontology edge weights and residual connections.
        """
        def __init__(self, units, attention_dropout=0.1, activation="relu", structural_bias_factor=1.0, 
                     use_residual=True, residual_weight=0.5, **kwargs):
            super().__init__(**kwargs)
            self.units = int(units)  # Force Python int to satisfy Keras 3 (rejects NumPy integers)
            self.attention_dropout = attention_dropout
            self.activation = activation
            self.structural_bias_factor = structural_bias_factor
            self.use_residual = use_residual
            self.residual_weight = residual_weight  # Format v1 uses the training model's 50/50 residual.

        def build(self, input_shape):
            self.node_transform = layers.Dense(int(self.units), use_bias=False, name="node_transform")

            # GATv2: Single attention vector applied AFTER the non-linearity
            self.attn_kernel = self.add_weight(
                name="attn_kernel",
                shape=(self.units, 1),
                initializer="glorot_uniform",
                trainable=True
            )

            # Structural bias weight as a trainable parameter
            self.structural_weight = self.add_weight(
                name="structural_weight",
                shape=(1,),
                initializer=tf.constant_initializer(self.structural_bias_factor),
                trainable=True
            )

            # Dropout layer
            self.dropout = layers.Dropout(self.attention_dropout)

            # Residual projection (handles input dim != output dim)
            if self.use_residual:
                self.residual_projection = layers.Dense(int(self.units), use_bias=False, name="residual_proj")

            super().build(input_shape)

        def call(self, inputs, training=None):
            """
            Applies GATv2 graph attention over node features with structural bias and residual connections.

            Args:
                inputs: tuple of (node_features, edges) or (node_features, edges, edge_weights)
                    - node_features: [num_nodes, feature_dim]
                    - edges: [num_edges, 2] where each row is [source_node_idx, target_node_idx]
                    - edge_weights: [num_edges, 1] with weights from ontology adjacency matrix
                training: boolean indicating whether in training mode

            Returns:
                Updated node features: [num_nodes, units]
            """
            # Handle both input formats for backward compatibility
            if len(inputs) == 3:
                node_states, edges, edge_weights = inputs
                use_structural_bias = True
            else:
                node_states, edges = inputs
                edge_weights = None
                use_structural_bias = False

            # Store original features for residual connection
            original_features = node_states

            # 1. Linear Transformation (W * h)
            # [num_nodes, units]
            node_states_transformed = self.node_transform(node_states)

            # 2. Gather Features for Edges
            # [num_edges, units]
            feat_src = tf.gather(node_states_transformed, edges[:, 0])
            feat_dst = tf.gather(node_states_transformed, edges[:, 1])

            # 3. GATv2 Operation: LeakyReLU(W*h_i + W*h_j)
            # Source and destination features use the same projection kernel.
            summed_features = feat_src + feat_dst
            activated_features = tf.nn.leaky_relu(summed_features, alpha=0.2)

            # 4. Compute Attention Scores: a^T * activated_features
            # [num_edges, 1]
            attention_logits = tf.matmul(activated_features, self.attn_kernel)

            # Apply structural bias from edge weights if available
            if use_structural_bias and edge_weights is not None:
                # Ensure edge_weights has proper shape for broadcasting
                if len(edge_weights.shape) == 1:
                    edge_weights = tf.expand_dims(edge_weights, -1)
                elif len(edge_weights.shape) == 2 and edge_weights.shape[1] != 1:
                    edge_weights = tf.expand_dims(edge_weights[:, 0], -1)

                # Scale edge weights by learnable factor and add to attention logits
                attention_logits += (edge_weights * self.structural_weight)

            # Clipping for numerical stability
            attention_logits = tf.clip_by_value(attention_logits, -20.0, 20.0)

            # 5. Softmax normalization by source node
            # Apply dropout to attention logits during training
            attention_logits = self.dropout(attention_logits, training=training)
            attention_scores = tf.exp(attention_logits)
            # Add clipping to prevent overflow in exponential
            attention_scores = tf.clip_by_value(attention_scores, 1e-10, 1e10)

            source_node_indices = edges[:, 0]
            source_node_sum = tf.math.unsorted_segment_sum(
                attention_scores,
                source_node_indices,
                num_segments=tf.shape(node_states)[0]
            )
            source_node_sum_for_edges = tf.gather(source_node_sum, source_node_indices)
            # Stabilize division for nodes with very small attention sums.
            attention_scores_norm = attention_scores / (source_node_sum_for_edges + 1e-6)

            # 6. Aggregation: Gather neighbor states and aggregate with attention weights
            node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
            # attention_scores_norm is [num_edges, 1], node_states_neighbors is [num_edges, units]
            aggregated = tf.math.unsorted_segment_sum(
                data=node_states_neighbors * attention_scores_norm,
                segment_ids=edges[:, 0],
                num_segments=tf.shape(node_states)[0],
            )

            # Apply activation function to aggregated features
            if self.activation:
                aggregated = tf.keras.activations.get(self.activation)(aggregated)

            # 7. Residual connection to preserve input diversity
            if self.use_residual:
                residual = self.residual_projection(original_features)
                # Blend aggregated features with the projected residual.
                out = (1.0 - self.residual_weight) * aggregated + self.residual_weight * residual
            else:
                out = aggregated

            return out

    class MultiHeadGraphAttention(layers.Layer):
        """Multi-head Graph Attention Network layer with structural bias"""
        def __init__(self, units, num_heads=Config.GAT_NUM_HEADS, merge_type="concat", dropout=Config.GAT_DROPOUT, 
                     attention_dropout=Config.GAT_ATTENTION_DROPOUT, activation=Config.GAT_ACTIVATION, 
                     structural_bias_factor=1.0, **kwargs):
            super().__init__(**kwargs)
            self.units = int(units)         # Force Python int — Keras 3 rejects NumPy integers
            self.num_heads = int(num_heads)  # Same guard for num_heads
            self.merge_type = merge_type
            self.dropout_rate = dropout
            self.attention_dropout = attention_dropout
            self.activation = activation
            self.structural_bias_factor = structural_bias_factor
            self.attention_layers = [
                GraphAttention(
                    units=units, 
                    attention_dropout=attention_dropout,
                    activation=None,  # Apply activation after merging heads
                    structural_bias_factor=structural_bias_factor
                ) for _ in range(num_heads)
            ]
            self.dropout = None  # Will be initialized in build
            self.layer_norm = None

        def build(self, input_shape):
            self.dropout = layers.Dropout(self.dropout_rate)
            # Initialize LayerNorm based on the expected output shape after merging heads
            if self.merge_type == "concat":
                 norm_shape = (input_shape[0][-1] * self.num_heads,) # Shape is [nodes, features * heads]
            else: # merge_type == "mean"
                 norm_shape = (input_shape[0][-1],) # Shape is [nodes, features]
            # Ensure norm_shape is correctly derived
            # If input_shape is a list/tuple (node_features, edges, edge_weights), use node_features shape
            node_feature_shape = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape

            if self.merge_type == "concat":
                 output_feature_dim = self.units * self.num_heads
            else: # merge_type == "mean"
                 output_feature_dim = self.units

            # LayerNorm applies normalization over the last dimension (features)
            self.layer_norm = layers.LayerNormalization(epsilon=1e-5) 

            super().build(input_shape)

        def call(self, inputs, training=None):
            # Store original node features for residual connection
            if isinstance(inputs, (list, tuple)):
                 original_node_features = inputs[0]
            else: # Use the input directly when it is not a feature/edge tuple.
                 original_node_features = inputs 

            # Handle both input formats for backward compatibility
            if len(inputs) == 3:
                node_features, edges, edge_weights = inputs
                use_structural_bias = True
            else:
                node_features, edges = inputs
                edge_weights = None
                use_structural_bias = False

            # Obtain outputs from each attention head
            if use_structural_bias:
                outputs = [
                    attention_layer([node_features, edges, edge_weights], training=training)
                    for attention_layer in self.attention_layers
                ]
            else:
                outputs = [
                    attention_layer([node_features, edges], training=training)
                    for attention_layer in self.attention_layers
                ]

            # Concatenate or average the node states from each head
            if self.merge_type == "concat":
                outputs = tf.concat(outputs, axis=-1)
            else:
                outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)

            # Apply dropout
            outputs = self.dropout(outputs, training=training)

            # Apply activation function after merging heads
            if self.activation:
                outputs = tf.keras.activations.get(self.activation)(outputs)

            # Add the residual only when the feature shapes match.
            if outputs.shape == original_node_features.shape:
                 outputs = outputs + original_node_features # Add residual
            else:
                 pass # Skip the residual for incompatible feature shapes.
            # Apply Layer Normalization
            outputs = self.layer_norm(outputs)

            return outputs

    class CellTypeEmbedder(layers.Layer):
        """GAT-based cell type embedder with ontology structure and semantic embeddings.

        Hierarchy edges use the adj[child, parent] = 1 convention:

        * ``ontology_adj_np`` / ``full_ontology_adj_np`` — GAT input graph
          (IS_A + semantic k-NN edges; not a hierarchy-only graph)
        * ``predictor.pure_ontology_adj`` / ``predictor.full_pure_ontology_adj``
          — IS_A hierarchy used for hierarchy traversal and visualization
        """
        def __init__(self, ontology_adj: np.ndarray, semantic_embeddings: Dict[str, np.ndarray],
                     classes: List[str], structural_dim: int = 50, full_ontology_adj: np.ndarray = None,
                     full_semantic_embeddings: Dict[str, np.ndarray] = None, full_classes: List[str] = None,
                     level_array: np.ndarray = None, full_level_array: np.ndarray = None, **kwargs):

            # Extract PPR/Prior parameters BEFORE calling super().__init__()
            # These are not recognized by the Keras base class
            self.prior_method = kwargs.pop('prior_method', 'ppr')
            self.ppr_alpha = kwargs.pop('ppr_alpha', 0.15)
            self.prior_matrix = kwargs.pop('prior_matrix', None)

            super().__init__(**kwargs)

            # Validate core inputs
            if not isinstance(ontology_adj, np.ndarray):
                raise TypeError(f"Expected ontology_adj to be np.ndarray, got {type(ontology_adj)}")
            if not isinstance(semantic_embeddings, dict):
                raise TypeError(f"Expected semantic_embeddings to be dict, got {type(semantic_embeddings)}")
            if len(classes) == 0:
                raise ValueError("Empty classes list provided to CellTypeEmbedder")

            # GAT input graph (IS_A + k-NN); not a clean hierarchy — see class docstring.
            self.ontology_adj_np = ontology_adj.astype(np.float32).copy()
            self.classes = list(classes)  # Make a copy of the list
            self.structural_dim = int(structural_dim)  # Force Python int — Keras 3 rejects NumPy integers
            self.semantic_embeddings_dict = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in semantic_embeddings.items()}
            # Full-universe GAT input graph (IS_A + k-NN); see class docstring.
            self.full_ontology_adj_np = full_ontology_adj.astype(np.float32).copy() if full_ontology_adj is not None else self.ontology_adj_np.copy()
            self.full_classes = list(full_classes) if full_classes is not None else list(self.classes)
            self.full_semantic_embeddings_dict = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in full_semantic_embeddings.items()} if full_semantic_embeddings is not None else self.semantic_embeddings_dict.copy()
            self.zero_shot_mode = full_ontology_adj is not None and full_classes is not None

            # --- Hierarchy Level Embeddings ---
            # Store level arrays for optional level embedding concatenation before GAT.
            # Level arrays map each class to its saved BFS depth in the ontology.
            if Config.USE_LEVEL_EMBEDDINGS and level_array is not None:
                # Validate shape consistency with class lists
                if level_array.shape[0] != len(classes):
                    raise ValueError(
                        f"level_array shape {level_array.shape[0]} does not match "
                        f"number of classes {len(classes)}"
                    )
                self.level_array_np = level_array.astype(np.int32).copy()

                if full_level_array is not None:
                    if full_level_array.shape[0] != len(self.full_classes):
                        raise ValueError(
                            f"full_level_array shape {full_level_array.shape[0]} does not match "
                            f"number of full_classes {len(self.full_classes)}"
                        )
                    self.full_level_array_np = full_level_array.astype(np.int32).copy()
                elif not self.zero_shot_mode:
                    # Non-zero-shot: full ontology == training ontology
                    self.full_level_array_np = self.level_array_np.copy()
                else:
                    # Fall back to training levels when full-ontology levels are absent.
                    self.full_level_array_np = self.level_array_np.copy()

                self.max_level = int(max(self.level_array_np.max(), self.full_level_array_np.max()))
                self.use_level_embeddings = True
            else:
                self.use_level_embeddings = False

            # Create edge lists from adjacency matrices for GAT (ensure copies)
            self.edges, self.edge_weights = self._create_edge_list(self.ontology_adj_np)
            self.edges = self.edges.copy()  # Independent array storage
            self.edge_weights = self.edge_weights.copy()  # Independent array storage

            if self.zero_shot_mode:
                self.full_edges, self.full_edge_weights = self._create_edge_list(self.full_ontology_adj_np)
                self.full_edges = self.full_edges.copy()  # Independent array storage
                self.full_edge_weights = self.full_edge_weights.copy()  # Independent array storage
            else:
                self.full_edges = self.edges.copy()  # Make a copy, not a reference
                self.full_edge_weights = self.edge_weights.copy()  # Make a copy, not a reference

            # Flag to determine if GAT embeddings should be trainable
            self.gat_trainable = Config.GAT_TRAINABLE
            # Initialize GAT layers in build().
            self.gat_layers = None
            self.full_gat_layers = None
            # Validate ontology matrix dimensions
            self._validate_ontology_shapes()
            # Complete missing semantic embeddings
            self._complete_semantic_embeddings()
            # Prepare semantic embedding arrays with validation
            self._prepare_semantic_arrays()

            # Transpose the graph for the reverse-direction attention pass.
            # Hierarchy edges change from (child, parent) to (parent, child).
            self.ontology_adj_down_np = self.ontology_adj_np.T
            self.edges_down, self.edge_weights_down = self._create_edge_list(self.ontology_adj_down_np)
            self.edges_down = self.edges_down.copy()
            self.edge_weights_down = self.edge_weights_down.copy()

            if self.zero_shot_mode:
                self.full_ontology_adj_down_np = self.full_ontology_adj_np.T
                self.full_edges_down, self.full_edge_weights_down = self._create_edge_list(self.full_ontology_adj_down_np)
                self.full_edges_down = self.full_edges_down.copy()
                self.full_edge_weights_down = self.full_edge_weights_down.copy()
            else:
                self.full_edges_down = self.edges_down.copy()
                self.full_edge_weights_down = self.edge_weights_down.copy()

        def _create_edge_list(self, adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Create edge list from adjacency matrix for GAT input, including edge weights"""
            edges = []
            edge_weights = []
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[1]):
                    if adj_matrix[i, j] > 0:
                        edges.append([i, j])
                        edge_weights.append([adj_matrix[i, j]])

            # If no edges, add self-loops
            if len(edges) == 0:
                edges = [[i, i] for i in range(adj_matrix.shape[0])]
                edge_weights = [[1.0] for _ in range(adj_matrix.shape[0])]
                print(f"[WARN] No edges found in ontology graph, adding {adj_matrix.shape[0]} self-loops")

            return np.array(edges, dtype=np.int32), np.array(edge_weights, dtype=np.float32)

        def _validate_ontology_shapes(self):
            """Validate ontology matrix dimensions match class counts"""
            train_shape = self.ontology_adj_np.shape
            if train_shape[0] != len(self.classes) or train_shape[1] != len(self.classes):
                raise ValueError(
                    f"Ontology adjacency matrix shape {train_shape} "
                    f"doesn't match class count {len(self.classes)}"
                )

            if self.zero_shot_mode:
                full_shape = self.full_ontology_adj_np.shape
                if full_shape[0] != len(self.full_classes) or full_shape[1] != len(self.full_classes):
                    raise ValueError(
                        f"Full ontology adjacency matrix shape {full_shape} "
                        f"doesn't match full class count {len(self.full_classes)}"
                    )

        def _find_ancestor_with_embedding(self, class_idx, ontology_adj, classes, embedding_dict, visited=None):
            """Find an embedding by traversing outgoing edges of the supplied graph.

            On an IS_A-only graph these edges lead to ancestors. GAT graphs
            can also contain semantic neighbors, so a match need not be an
            ontology ancestor.

            Args:
                class_idx: Index of the class to find an ancestor for
                ontology_adj: Adjacency matrix representing the ontology
                classes: List of class names corresponding to matrix indices
                embedding_dict: Dictionary of existing embeddings
                visited: Set of already visited indices to avoid cycles

            Returns:
                A copy of the first reachable embedding, or None if none is found.
            """
            if visited is None:
                visited = set()

            # Avoid cycles
            if class_idx in visited:
                return None
            visited.add(class_idx)

            # Follow outgoing edges, excluding the current node.
            parent_indices = []
            for j in range(ontology_adj.shape[1]):
                if ontology_adj[class_idx, j] > 0 and j != class_idx:  # Exclude self-loops
                    parent_indices.append(j)

            # Prefer a directly connected node with an embedding.
            for parent_idx in parent_indices:
                parent_class = classes[parent_idx]
                # Copy the matching node's embedding.
                if parent_class in embedding_dict:
                    return embedding_dict[parent_class].copy()

            # If no direct neighbor has an embedding, traverse recursively.
            for parent_idx in parent_indices:
                ancestor_embedding = self._find_ancestor_with_embedding(
                    parent_idx, ontology_adj, classes, embedding_dict, visited
                )
                if ancestor_embedding is not None:
                    return ancestor_embedding

            # No reachable embedding was found.
            return None

        def _complete_semantic_embeddings(self):
            """Fill missing embeddings from reachable graph nodes, with random fallback."""
            # Handle training classes
            missing_train = [cls for cls in self.classes if cls not in self.semantic_embeddings_dict]
            if missing_train:
                emb_dim = next(iter(self.semantic_embeddings_dict.values())).shape[0]
                print(f"[WARN] Missing semantic embeddings for {len(missing_train)} training classes. "
                      f"Using ancestor embeddings when available, otherwise random vectors. First few: {missing_train[:5]}")

                # For each missing class, search the supplied graph for an embedding.
                imputed_count = 0
                random_count = 0
                for cls_idx, cls in enumerate(self.classes):
                    if cls in missing_train:
                        # Search outgoing graph edges for an available embedding.
                        ancestor_embedding = self._find_ancestor_with_embedding(
                            cls_idx, self.ontology_adj_np, self.classes, self.semantic_embeddings_dict
                        )

                        if ancestor_embedding is not None:
                            # Use the reachable node's embedding.
                            self.semantic_embeddings_dict[cls] = ancestor_embedding
                            imputed_count += 1
                        else:
                            # Fall back to a random vector when no reachable embedding exists.
                            self.semantic_embeddings_dict[cls] = np.random.normal(0, 0.1, emb_dim).astype(np.float32)
                            random_count += 1

                print(f"[INFO]   Completed training embeddings: {imputed_count} from ancestors, {random_count} with random init.")

            # Handle full ontology classes if in zero-shot mode
            if self.zero_shot_mode:
                missing_full = [cls for cls in self.full_classes if cls not in self.full_semantic_embeddings_dict]
                if missing_full:
                    emb_dim = next(iter(self.full_semantic_embeddings_dict.values())).shape[0]
                    print(f"[WARN] Missing semantic embeddings for {len(missing_full)} full ontology classes. "
                          f"Using ancestor embeddings when available, otherwise random vectors. First few: {missing_full[:5]}")

                    # For each missing class, search the supplied graph for an embedding.
                    imputed_count = 0
                    random_count = 0
                    for cls_idx, cls in enumerate(self.full_classes):
                        if cls in missing_full:
                            # Search outgoing graph edges for an available embedding.
                            ancestor_embedding = self._find_ancestor_with_embedding(
                                cls_idx, self.full_ontology_adj_np, self.full_classes, self.full_semantic_embeddings_dict
                            )

                            if ancestor_embedding is not None:
                                # Use the reachable node's embedding.
                                self.full_semantic_embeddings_dict[cls] = ancestor_embedding
                                imputed_count += 1
                            else:
                                # Fall back to a random vector when no reachable embedding exists.
                                self.full_semantic_embeddings_dict[cls] = np.random.normal(0, 0.1, emb_dim).astype(np.float32)
                                random_count += 1

                    print(f"[INFO]   Completed full ontology embeddings: {imputed_count} from ancestors, {random_count} with random init.")

        def _prepare_semantic_arrays(self):
            """Prepare normalized semantic arrays for the training and full ontologies."""
            # Create the training-ontology semantic array.
            self._create_hybrid_array(self.classes, self.semantic_embeddings_dict, 'training')
            # Create the full-ontology semantic array in zero-shot mode.
            if self.zero_shot_mode:
                self._create_hybrid_array(self.full_classes, self.full_semantic_embeddings_dict, 'full ontology')

        def _create_hybrid_array(self, classes, embeddings, name):
            """Stack and normalize semantic vectors for GAT input.

            Feature width comes from the supplied embeddings and is independent
            of the number of ontology classes.
            """
            try:
                num_classes = len(classes)

                # Stack semantic embeddings in class order.
                semantic_array = np.stack([embeddings[cls] for cls in classes], axis=0).astype(np.float32)

                # Check for NaN or Inf values
                if np.any(np.isnan(semantic_array)) or np.any(np.isinf(semantic_array)):
                    print(f"[WARN] Found NaN or Inf values in {name} semantic embeddings. Replacing with zeros.")
                    semantic_array = np.nan_to_num(semantic_array)

                # Normalize semantic embeddings for stable training
                semantic_normalized = semantic_array / (np.linalg.norm(semantic_array, axis=1, keepdims=True) + 1e-8)

                hybrid_array = semantic_normalized

                # Calculate meaningful diversity metrics
                # 1. Pairwise cosine distances (measures semantic separation)
                from sklearn.metrics.pairwise import cosine_similarity
                cos_sim = cosine_similarity(semantic_normalized)
                # Exclude diagonal (self-similarity)
                mask = ~np.eye(cos_sim.shape[0], dtype=bool)
                pairwise_similarities = cos_sim[mask]
                mean_similarity = np.mean(pairwise_similarities)
                std_similarity = np.std(pairwise_similarities)

                # 2. Mean pairwise distance (1 - cosine similarity)
                mean_distance = 1.0 - mean_similarity

                # 3. Effective rank (measures how many "independent" directions are used)
                # Based on entropy of singular values
                _, s, _ = np.linalg.svd(semantic_normalized, full_matrices=False)
                s_normalized = s / np.sum(s)
                entropy = -np.sum(s_normalized * np.log(s_normalized + 1e-10))
                effective_rank = np.exp(entropy)

                # Store the array with the appropriate name
                if name == 'training':
                    self.semantic_emb_array = hybrid_array
                else:
                    self.full_semantic_emb_array = hybrid_array

            except Exception as e:
                print(f"[ERROR] Failed to create {name} semantic embeddings: {e}")
                traceback.print_exc()
                # Fallback: Create random embeddings with fixed dimension
                num_classes = len(classes)
                emb_dim = next(iter(embeddings.values())).shape[0]
                fallback_array = np.random.normal(0, 0.1, (num_classes, emb_dim)).astype(np.float32)

                if name == 'training':
                    self.semantic_emb_array = fallback_array
                else:
                    self.full_semantic_emb_array = fallback_array
                print(f"[WARN] Using fallback random embeddings for {name} with shape {fallback_array.shape}")

        def build(self, input_shape):
            """Build the CellTypeEmbedder layer by initializing Bi-Directional GAT layers"""
            super().build(input_shape)

            # Initialize Bi-Directional GAT layers
            self.gat_layers_up = []
            self.gat_layers_down = []
            self.gat_projections_up = []
            self.gat_projections_down = []

            for i in range(Config.GAT_NUM_LAYERS):
                # Calculate units per head for concatenation to maintain structural_dim
                # If merge_type is 'mean', units should be structural_dim
                # If merge_type is 'concat', units should be structural_dim // num_heads
                is_last_layer = (i == Config.GAT_NUM_LAYERS - 1)

                # Determine merge type and units
                if Config.GAT_NUM_HEADS > 1:
                    merge_type = "concat"
                    units_per_head = self.structural_dim // Config.GAT_NUM_HEADS
                else:
                    merge_type = "mean"
                    units_per_head = self.structural_dim

                # Upward GAT (Child -> Parent)
                gat_up = MultiHeadGraphAttention(
                    units=units_per_head,
                    num_heads=Config.GAT_NUM_HEADS,
                    merge_type=merge_type,
                    attention_dropout=Config.GAT_ATTENTION_DROPOUT,
                    dropout=Config.GAT_DROPOUT,
                    activation=Config.GAT_ACTIVATION,
                    structural_bias_factor=1.0,
                    name=f"gat_layer_up_{i}"
                )
                self.gat_layers_up.append(gat_up)

                # Downward GAT (Parent -> Child)
                gat_down = MultiHeadGraphAttention(
                    units=units_per_head,
                    num_heads=Config.GAT_NUM_HEADS,
                    merge_type=merge_type,
                    attention_dropout=Config.GAT_ATTENTION_DROPOUT,
                    dropout=Config.GAT_DROPOUT,
                    activation=Config.GAT_ACTIVATION,
                    structural_bias_factor=1.0,
                    name=f"gat_layer_down_{i}"
                )
                self.gat_layers_down.append(gat_down)

                # Check if projection is needed (if structural_dim is not divisible by num_heads)
                self.gat_output_dim = units_per_head * Config.GAT_NUM_HEADS if merge_type == "concat" else units_per_head
                self.use_projection = (self.gat_output_dim != self.structural_dim)
                if self.use_projection:
                    self.gat_projections_up.append(layers.Dense(self.structural_dim, name=f"projection_up_{i}"))
                    self.gat_projections_down.append(layers.Dense(self.structural_dim, name=f"projection_down_{i}"))
                else:
                    self.gat_projections_up.append(None)
                    self.gat_projections_down.append(None)

            # Learn a gate balancing upward and downward graph features.
            self.fusion_gate = layers.Dense(1, activation='sigmoid', name="fusion_gate")

            # Share layers for zero-shot mode
            if self.zero_shot_mode:
                self.full_gat_layers_up = self.gat_layers_up
                self.full_gat_layers_down = self.gat_layers_down
            else:
                self.full_gat_layers_up = self.gat_layers_up
                self.full_gat_layers_down = self.gat_layers_down

            # Keep graph tensors as persistent variables to avoid rebuilding
            # constants on every call.

            # Training Ontology Tensors
            self.edges_up_var = self.add_weight(
                name="edges_up",
                shape=self.edges.shape,
                dtype=tf.int64,
                initializer=tf.constant_initializer(self.edges),
                trainable=False
            )
            self.edge_weights_up_var = self.add_weight(
                name="edge_weights_up",
                shape=self.edge_weights.shape,
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.edge_weights),
                trainable=False
            )
            self.edges_down_var = self.add_weight(
                name="edges_down",
                shape=self.edges_down.shape,
                dtype=tf.int64,
                initializer=tf.constant_initializer(self.edges_down),
                trainable=False
            )
            self.edge_weights_down_var = self.add_weight(
                name="edge_weights_down",
                shape=self.edge_weights_down.shape,
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.edge_weights_down),
                trainable=False
            )
            self.semantic_emb_var = self.add_weight(
                name="semantic_emb",
                shape=self.semantic_emb_array.shape,
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.semantic_emb_array),
                trainable=False
            )

            # --- Hierarchy Level Embedding Layer and Weight Tensors ---
            if self.use_level_embeddings:
                # Learnable embedding layer mapping integer BFS levels to dense vectors
                self.level_embedding_layer = tf.keras.layers.Embedding(
                    input_dim=self.max_level + 1,
                    output_dim=Config.LEVEL_EMBEDDING_DIM,
                    name="level_embedding"
                )

                # Non-trainable level ID tensor for training ontology classes
                self.level_var = self.add_weight(
                    name="level_ids",
                    shape=self.level_array_np.shape,
                    dtype=tf.int32,
                    initializer=tf.constant_initializer(self.level_array_np),
                    trainable=False
                )

                # Non-trainable level ID tensor for full ontology (zero-shot mode only)
                if self.zero_shot_mode:
                    self.full_level_var = self.add_weight(
                        name="full_level_ids",
                        shape=self.full_level_array_np.shape,
                        dtype=tf.int32,
                        initializer=tf.constant_initializer(self.full_level_array_np),
                        trainable=False
                    )

            # Full Ontology Tensors (if zero-shot mode)
            if self.zero_shot_mode:
                self.full_edges_up_var = self.add_weight(
                    name="full_edges_up",
                    shape=self.full_edges.shape,
                    dtype=tf.int64,
                    initializer=tf.constant_initializer(self.full_edges),
                    trainable=False
                )
                self.full_edge_weights_up_var = self.add_weight(
                    name="full_edge_weights_up",
                    shape=self.full_edge_weights.shape,
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(self.full_edge_weights),
                    trainable=False
                )
                self.full_edges_down_var = self.add_weight(
                    name="full_edges_down",
                    shape=self.full_edges_down.shape,
                    dtype=tf.int64,
                    initializer=tf.constant_initializer(self.full_edges_down),
                    trainable=False
                )
                self.full_edge_weights_down_var = self.add_weight(
                    name="full_edge_weights_down",
                    shape=self.full_edge_weights_down.shape,
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(self.full_edge_weights_down),
                    trainable=False
                )
                self.full_semantic_emb_var = self.add_weight(
                    name="full_semantic_emb",
                    shape=self.full_semantic_emb_array.shape,
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(self.full_semantic_emb_array),
                    trainable=False
                )

        def call(self, inputs, training=None):
            """Return unnormalized training-ontology prototypes; downstream heads normalize them."""
            # Pass training parameter to get_gat_prototypes for proper dropout behavior
            return self.get_gat_prototypes(for_full_ontology=False, normalize=False, training=training)

        def get_gat_prototypes(self, for_full_ontology=False, normalize=True, training=False):
            """Compute GAT-derived class prototypes for the training or full ontology.

            Runs the relational GAT forward pass over the ontology graph and
            returns one prototype vector per class.

            Args:
                for_full_ontology: Whether to use full ontology (zero-shot mode)
                normalize: Whether to L2 normalize prototypes (default: True)
                training: Whether to apply dropout (default: False for inference)

            Returns:
                Class prototype tensor [num_classes, structural_dim]
            """
            if for_full_ontology and self.zero_shot_mode:
                # Use full ontology data (from persistent variables)
                edges_up = self.full_edges_up_var
                edge_weights_up = self.full_edge_weights_up_var
                edges_down = self.full_edges_down_var
                edge_weights_down = self.full_edge_weights_down_var
                semantic_emb = self.full_semantic_emb_var
                gat_layers_up = self.full_gat_layers_up
                gat_layers_down = self.full_gat_layers_down
            else:
                # Use training ontology data (from persistent variables)
                edges_up = self.edges_up_var
                edge_weights_up = self.edge_weights_up_var
                edges_down = self.edges_down_var
                edge_weights_down = self.edge_weights_down_var
                semantic_emb = self.semantic_emb_var
                gat_layers_up = self.gat_layers_up
                gat_layers_down = self.gat_layers_down

            # Use raw semantic embeddings directly as GAT input.
            h = tf.cast(semantic_emb, self.compute_dtype)

            # Concatenate learnable level embeddings with semantic features before GAT.
            # Level embeddings encode BFS depth in the ontology hierarchy, giving the
            # GAT explicit awareness of structural depth (e.g. [N, 768] -> [N, 800]).
            if self.use_level_embeddings:
                level_ids = self.full_level_var if (for_full_ontology and self.zero_shot_mode) else self.level_var
                level_emb = self.level_embedding_layer(tf.cast(level_ids, tf.int32))
                level_emb = tf.cast(level_emb, h.dtype)
                h = tf.concat([h, level_emb], axis=-1)

            # 2. RELATIONAL GAT PASS
            for i in range(len(gat_layers_up)):
                # Attend over the graph with (child, parent) hierarchy edges.
                h_up = self.gat_layers_up[i]((h, edges_up, edge_weights_up), training=training)

                # Attend over the transposed graph with (parent, child) hierarchy edges.
                h_down = self.gat_layers_down[i]((h, edges_down, edge_weights_down), training=training)

                # Apply dimension projection if needed
                if hasattr(self, 'gat_projections_up') and self.gat_projections_up[i] is not None:
                    h_up = self.gat_projections_up[i](h_up)

                if hasattr(self, 'gat_projections_down') and self.gat_projections_down[i] is not None:
                    h_down = self.gat_projections_down[i](h_down)

                # Values near 1 favor upward features; values near 0 favor
                # downward features.
                gate = self.fusion_gate(tf.concat([h_up, h_down], axis=-1))

                # Blend the two directional representations.
                h_fused = gate * h_up + (1 - gate) * h_down

                # Skip connection (Residual) — only when shapes are compatible.
                # On the first GAT layer, h may be [N, 768+32] (input) while h_fused
                # is [N, structural_dim], so we skip the residual in that case.
                # GraphAttention already has its own internal residual projection.
                if h.shape[-1] == h_fused.shape[-1]:
                    h = h_fused + h
                else:
                    h = h_fused

            # Normalize prototypes when requested.
            if normalize:
                h = tf.nn.l2_normalize(h, axis=-1, epsilon=1e-8)

            return h

        def get_ppr_matrix(self, for_full_ontology: bool = False, alpha: float = 0.15) -> np.ndarray:
            """
            Compute Personalized PageRank (PPR) matrix on the fly.

            This enables Profile-Based Voting during inference by providing the "Theoretical Profile"
            (topological fingerprints) for all classes.

            Args:
                for_full_ontology: Whether to compute for full ontology or training ontology
                alpha: Restart probability (default: 0.15)

            Returns:
                PPR Matrix [num_classes, num_classes]
            """
            # Select appropriate adjacency matrix
            if for_full_ontology:
                adj = self.full_ontology_adj_np
                # Ensure we have the full graph
                if adj is None:
                    print("[WARN] Full ontology adjacency not available, falling back to training ontology")
                    adj = self.ontology_adj_np
            else:
                adj = self.ontology_adj_np

            num_classes = adj.shape[0]

            # 1. Symmetrize the adjacency for the similarity-profile calculation.
            adj_sym = np.maximum(adj, adj.T)

            # Add self-loops
            adj_sym = adj_sym + np.eye(num_classes)

            # 2. Compute Normalized Adjacency: D^(-1/2) * A * D^(-1/2)
            degrees = np.sum(adj_sym, axis=1)
            degrees = np.maximum(degrees, 1e-8)
            d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
            norm_adj = d_inv_sqrt @ adj_sym @ d_inv_sqrt

            # 3. Compute PPR: P = alpha * (I - (1-alpha) * A_norm)^-1
            identity = np.eye(num_classes)

            try:
                matrix_to_invert = identity - (1 - alpha) * norm_adj
                inv_matrix = np.linalg.inv(matrix_to_invert)
                prior_matrix = alpha * inv_matrix

                # Row-normalize
                row_sums = prior_matrix.sum(axis=1, keepdims=True)
                prior_matrix = prior_matrix / np.maximum(row_sums, 1e-8)

                return prior_matrix.astype(np.float32)

            except np.linalg.LinAlgError:
                print("[WARN] PPR Inversion failed in CellTypeEmbedder, returning identity")
                return np.eye(num_classes, dtype=np.float32)

    class EdgeIndexGAT(tf.keras.layers.Layer):
        """
        Graph Attention Network layer that works directly with edge indices.

        Uses additive attention with LeakyReLU without a dense adjacency matrix,
        allowing batch query nodes to attend to memory-bank context nodes.

        Based on Veličković et al. (2017) "Graph Attention Networks"
        but adapted to work with edge_index format like PyTorch Geometric.

        Self-loop convention
        --------------------
        By default the caller supplies all edges, including any self-loops.
        Set ``add_self_loops=True`` to append ``(i,i)`` for every query node
        ``i in [0, batch_size)``. Edges remain grouped into equal-sized query
        blocks. Match the checkpoint's neighbor convention when constructing
        the graph, and avoid enabling this option if self-loops are already
        supplied.
        """

        def __init__(
            self,
            units,
            num_heads=1,
            concat_heads=True,
            dropout_rate=0.0,
            activation='relu',
            use_bias=True,
            kernel_regularizer=None,
            add_self_loops: bool = False,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.units = int(units)           # Force Python int — Keras 3 rejects NumPy integers
            self.num_heads = int(num_heads)    # Same guard for num_heads
            self.concat_heads = concat_heads
            self.dropout_rate = dropout_rate
            self.activation = tf.keras.activations.get(activation)
            self.use_bias = use_bias
            self.kernel_regularizer = kernel_regularizer
            self.add_self_loops = bool(add_self_loops)

            # Calculate output dimension
            if concat_heads:
                self.output_dim = self.units * self.num_heads
            else:
                self.output_dim = self.units

        def build(self, input_shape):
            # input_shape is [node_features_shape, edge_index_shape]
            node_features_shape = input_shape[0]
            input_dim = node_features_shape[-1]

            # Feature transformation weights (shared across heads)
            self.kernel = self.add_weight(
                name='kernel',
                shape=(input_dim, self.units * self.num_heads),
                initializer='glorot_uniform',
                regularizer=self.kernel_regularizer,
                trainable=True
            )

            # Attention mechanism weights (one per head)
            self.attn_kernel_self = self.add_weight(
                name='attn_kernel_self',
                shape=(self.num_heads, self.units),
                initializer='glorot_uniform',
                trainable=True
            )

            self.attn_kernel_neigh = self.add_weight(
                name='attn_kernel_neigh',
                shape=(self.num_heads, self.units),
                initializer='glorot_uniform',
                trainable=True
            )

            if self.use_bias:
                self.bias = self.add_weight(
                    name='bias',
                    shape=(self.output_dim,),
                    initializer='zeros',
                    trainable=True
                )

            super().build(input_shape)

        def call(self, inputs, training=None, batch_size=None):
            """
            Forward pass.

            Args:
                inputs: Tuple of (node_features, edge_index)
                    - node_features: [N_total, F] (batch + memory)
                    - edge_index: [2, E] where E = batch_size * k. Edges must
                      be grouped by query row, with the same k for each query.
                training: Whether in training mode
                batch_size: Number of target nodes to update (rest are context)

            Returns:
                Updated features [batch_size, output_dim]
            """
            node_features, edge_index = inputs

            # Linear transformation: [N, F] @ [F, units*heads] -> [N, units*heads]
            features_transformed = tf.matmul(node_features, self.kernel)

            # Reshape to [N, heads, units]
            N = tf.shape(features_transformed)[0]
            features_transformed = tf.reshape(
                features_transformed,
                [N, self.num_heads, self.units]
            )

            # Determine batch size (nodes to update)
            if batch_size is None:
                batch_size_val = N
            else:
                batch_size_val = batch_size

            # Optionally append (i, i) for every query node i so the layer is
            # self-loop-canonical when the caller's k-NN doesn't include the
            # query in its own neighbor list (e.g., anchors-only candidates).
            # Self-loops are appended PER NODE so the per-node block layout
            # required by the downstream reshape ([batch_size, k, ...]) survives.
            if self.add_self_loops:
                sources_2d = tf.reshape(edge_index[0], [batch_size_val, -1])
                targets_2d = tf.reshape(edge_index[1], [batch_size_val, -1])
                self_idx = tf.range(batch_size_val, dtype=edge_index.dtype)[:, None]
                sources_2d = tf.concat([sources_2d, self_idx], axis=1)
                targets_2d = tf.concat([targets_2d, self_idx], axis=1)
                edge_index = tf.stack(
                    [tf.reshape(sources_2d, [-1]),
                     tf.reshape(targets_2d, [-1])],
                    axis=0,
                )

            # Extract source and target indices from edge_index
            edge_sources = edge_index[0]  # Batch node indices (repeated)
            edge_targets = edge_index[1]  # Neighbor indices in combined features

            # Gather source and target features
            # sources: [E, heads, units]
            # targets: [E, heads, units]
            features_source = tf.gather(features_transformed, edge_sources)
            features_target = tf.gather(features_transformed, edge_targets)

            # Compute attention logits (additive mechanism)
            # attn_for_source: [E, heads]
            # attn_for_target: [E, heads]
            attn_for_source = tf.reduce_sum(
                features_source * self.attn_kernel_self,  # Broadcasting
                axis=-1
            )
            attn_for_target = tf.reduce_sum(
                features_target * self.attn_kernel_neigh,
                axis=-1
            )

            # Combine and apply LeakyReLU
            attn_logits = tf.nn.leaky_relu(attn_for_source + attn_for_target, alpha=0.2)

            # Apply dropout to attention logits during training.
            if training and self.dropout_rate > 0:
                attn_logits = tf.nn.dropout(attn_logits, rate=self.dropout_rate)

            # Reshape edge indices to [batch_size, k] to enable per-node softmax
            num_edges = tf.shape(edge_index)[1]
            k = num_edges // batch_size_val

            # Reshape attention logits: [E, heads] -> [batch_size, k, heads]
            attn_logits_reshaped = tf.reshape(attn_logits, [batch_size_val, k, self.num_heads])

            # Apply softmax per node (over its k neighbors)
            attn_coeffs = tf.nn.softmax(attn_logits_reshaped, axis=1)  # [batch_size, k, heads]

            # Reshape target features: [E, heads, units] -> [batch_size, k, heads, units]
            features_target_reshaped = tf.reshape(
                features_target,
                [batch_size_val, k, self.num_heads, self.units]
            )

            # Apply attention: [batch_size, k, heads, 1] * [batch_size, k, heads, units]
            attn_coeffs_expanded = tf.expand_dims(attn_coeffs, axis=-1)  # [batch_size, k, heads, 1]
            weighted_features = attn_coeffs_expanded * features_target_reshaped

            # Aggregate: sum over neighbors (k dimension)
            aggregated = tf.reduce_sum(weighted_features, axis=1)  # [batch_size, heads, units]

            # Combine heads
            if self.concat_heads:
                # Concatenate: [batch_size, heads * units]
                output = tf.reshape(aggregated, [batch_size_val, self.num_heads * self.units])
            else:
                # Average: [batch_size, units]
                output = tf.reduce_mean(aggregated, axis=1)

            # Apply bias
            if self.use_bias:
                output = output + self.bias

            # Apply activation
            if self.activation is not None:
                output = self.activation(output)

            # Apply dropout to output
            if training and self.dropout_rate > 0:
                output = tf.nn.dropout(output, rate=self.dropout_rate)

            return output

    class EdgeIndexTransformer(tf.keras.layers.Layer):
        """
        Graph Transformer layer with dot-product attention that works with edge indices.

        Uses a pre-norm residual architecture:
        Output = Activation(x + Dropout(Attention(LayerNorm(x))) + optional_bias)

        """
        def __init__(self, units, num_heads, dropout_rate=0.0, activation='relu', use_bias=True, **kwargs):
            super().__init__(**kwargs)
            self.units = int(units)           # Force Python int — Keras 3 rejects NumPy integers
            self.num_heads = int(num_heads)    # Same guard for num_heads
            self.dropout_rate = dropout_rate
            self.activation_name = activation
            self.activation_fn = tf.keras.activations.get(activation)
            self.use_bias = use_bias

            # Adjust num_heads to ensure units is evenly divisible by num_heads
            # This prevents dimension mismatch errors in MultiHeadAttention
            effective_num_heads = self.num_heads

            while self.units % effective_num_heads != 0 and effective_num_heads > 1:
                effective_num_heads -= 1

            self.effective_num_heads = effective_num_heads



            # Components
            self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

            # Calculate key_dim ensuring it's reasonable for GPU kernels
            key_dim = units // self.effective_num_heads

            # With no output_shape override, MHA projects back to query width.
            # The residual and optional bias require that width to equal units.
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads=self.effective_num_heads,
                key_dim=key_dim,
                dropout=dropout_rate
            )
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
            self.add = tf.keras.layers.Add()

            if use_bias:
                self.bias = None  # Will be created in build

        def build(self, input_shape):
            if self.use_bias:
                self.bias = self.add_weight(name='bias', shape=(self.units,), initializer='zeros')
            super().build(input_shape)

        def call(self, inputs, training=None, batch_size=None):
            """Update the first batch_size query rows from grouped neighbour blocks.

            inputs contains node_features [n_nodes, feature_width] and
            edge_index [2, batch_size*k]. Neighbour indices in edge_index[1]
            must contain k consecutive entries for each query, in query order.
            With batch_size=None, all feature rows are queries. Returns
            [batch_size, feature_width]; an enabled bias requires feature_width
            to equal units.
            """
            node_features, edge_index = inputs

            # Determine batch size
            if batch_size is not None:
                batch_size_val = batch_size
            else:
                batch_size_val = tf.shape(node_features)[0]

            # 1. Normalize all node features before attention.
            # Normalize ALL features before gathering to ensure consistent scale
            node_features_norm = self.layernorm(node_features)

            # 2. Prepare Query (Target nodes)
            # h_query: Original features for residual connection [batch, F]
            h_query = node_features[:batch_size_val]
            # h_query_norm: Normalized features for attention [batch, F]
            h_query_norm = node_features_norm[:batch_size_val]

            # 3. Prepare Key/Value (Neighbor nodes)
            # Determine number of neighbors (k)
            num_edges = tf.shape(edge_index)[1]
            k = num_edges // batch_size_val

            # Gather neighbors from NORMALIZED features
            neighbor_indices = edge_index[1]
            h_neighbors_norm = tf.reshape(
                tf.gather(node_features_norm, neighbor_indices), 
                [batch_size_val, k, -1]
            )

            # 4. Multi-Head Attention
            # query: [batch, 1, features]
            # value/key: [batch, k, features]
            h_query_exp = tf.expand_dims(h_query_norm, 1)

            attn_out = self.mha(
                query=h_query_exp, 
                value=h_neighbors_norm, 
                key=h_neighbors_norm, 
                training=training
            )

            # Squeeze back to [batch, features]
            attn_out = tf.squeeze(attn_out, 1)

            # 5. Dropout on Attention Output
            if training:
                attn_out = self.dropout(attn_out)

            # 6. Add the query residual.
            # Output = x + Attention(Norm(x))
            output = self.add([h_query, attn_out])

            # 7. Output Bias and Activation
            if self.use_bias and self.bias is not None:
                output = output + self.bias

            if self.activation_fn:
                output = self.activation_fn(output)

            return output

    class VariationalHead(tf.keras.layers.Layer):
        """Variational projection head used by the dual-head encoder."""

        def __init__(self, latent_dim: int, name: str, **kwargs):
            super().__init__(name=name, **kwargs)
            self.latent_dim = latent_dim
            self.head_name = name
            self.dense_mu = tf.keras.layers.Dense(latent_dim, activation=None, name=f'{name}_mu')
            self.dense_log_var = tf.keras.layers.Dense(latent_dim, activation=None, name=f'{name}_log_var')

        def call(
            self,
            shared_features: tf.Tensor,
            training: Optional[bool] = None,
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            mu = self.dense_mu(shared_features)
            log_var = tf.clip_by_value(self.dense_log_var(shared_features), -10.0, 10.0)

            if training is None:
                training_tensor = tf.constant(False)
            else:
                training_tensor = tf.cast(training, tf.bool)

            epsilon = tf.random.normal(shape=tf.shape(mu), dtype=mu.dtype)
            z_sampled = mu + tf.exp(0.5 * log_var) * epsilon
            z = tf.where(training_tensor, z_sampled, mu)
            return z, mu, log_var

    class DualHeadVGAEEncoder(tf.keras.Model):
        """
        Gene-token VGAE encoder with reconstruction and classification heads.

        The shared encoder applies the gene-token front-end, graph attention,
        dropout, and layer normalization. Each variational head returns sampled
        embeddings, means, and log variances. Checkpoint gene order defines the
        external input columns.
        """

        def __init__(
            self,
            num_genes: int,
            latent_dim: int = 512,
            hidden_dims: Optional[List[int]] = None,
            shared_dim: int = 1024,
            num_attention_heads: int = 4,
            dropout_rate: float = 0.3,
            activation: str = "relu",
            use_bias: bool = True,
            kernel_regularizer: Optional[str] = None,
            numerical_stability_eps: float = 1e-8,
            graph_attention_type: str = "transformer",
            use_gene_tokens: bool = False,
            gene_token_table=None,
            gene_token_col_to_row=None,
            gene_token_k: int = 2048,
            gene_token_dim: int = 1024,
            gene_token_heads: int = 8,
            **kwargs,
        ):
            """
            Initialize dual-head VGAE encoder.

            Args:
                num_genes: Number of genes in vocabulary (input dimension)
                latent_dim: Latent embedding dimension for each head
                hidden_dims: Compatibility metadata; its final value sets shared_dim
                            when supplied. This package does not build dense hidden layers.
                shared_dim: Dimension of shared encoder output (before heads split).
                           When hidden_dims is supplied, its last entry overrides this value.
                num_attention_heads: Number of attention heads for attention mechanism
                dropout_rate: Dropout rate for regularization
                activation: Activation function for graph attention.
                use_bias: Whether to use bias in layers
                kernel_regularizer: Kernel regularizer ('l1', 'l2', or None)
                numerical_stability_eps: Epsilon for numerical stability
                graph_attention_type: Type of graph attention ('gat' or 'transformer')
                    - 'gat': Additive attention
                    - 'transformer': Dot-product attention
                use_gene_tokens: Must be True for the supported gene-token front-end.
                gene_token_table: Frozen protein-embedding table [R_prot+1, emb_dim]
                    (checkpoint shared_encoder weight_1) — required when
                    use_gene_tokens=True.
                gene_token_col_to_row: Frozen (species, column) -> table-row map
                    [num_species, num_genes] int (checkpoint shared_encoder weight_0).
                gene_token_k: Top-k genes kept per cell (default 2048).
                gene_token_dim: Per-gene token dimension (default 1024).
                gene_token_heads: Pooling attention heads in the front-end (default 8).
            """
            super().__init__(**kwargs)

            # Store parameters
            self.num_genes = num_genes
            self.latent_dim = latent_dim
            self.hidden_dims = hidden_dims if hidden_dims is not None else []

            # hidden_dims supplies shared-width metadata; no dense hidden stack is built.
            if self.hidden_dims:
                self.shared_dim = self.hidden_dims[-1]
                if shared_dim != self.hidden_dims[-1]:
                    print(f"[INFO] DualHeadVGAEEncoder: Auto-adjusting shared_dim from {shared_dim} to {self.shared_dim} (last hidden layer dimension)")
            else:
                self.shared_dim = shared_dim

            self.num_attention_heads = num_attention_heads
            self.dropout_rate = dropout_rate
            self.activation = activation
            self.use_bias = use_bias
            self.eps = numerical_stability_eps
            self.graph_attention_type = graph_attention_type

            # Input remains num_genes-wide; encode_hidden appends the
            # per-batch checkpoint species index internally.
            self.use_gene_tokens = bool(use_gene_tokens)
            self._gene_token_table = gene_token_table
            self._gene_token_col_to_row = gene_token_col_to_row
            self.gene_token_k = int(gene_token_k)
            self.gene_token_dim = int(gene_token_dim)
            self.gene_token_heads = int(gene_token_heads)
            # species_id indexes checkpoint species_names; callers set it per batch.
            self.species_id = 0

            # Configure regularizer
            if kernel_regularizer == "l1":
                self.regularizer = tf.keras.regularizers.L1(0.01)
            elif kernel_regularizer == "l2":
                self.regularizer = tf.keras.regularizers.L2(0.01)
            else:
                self.regularizer = None

            # Direct processing: input dimension is number of genes
            self.effective_input_dim = num_genes

            # Build encoder components
            self._build_encoder_layers()



        @property
        def input_dim(self) -> int:
            """External gene-input dimension."""
            return self.num_genes

        def build(self, input_shape):
            """
            Build all sublayers with a four-node fully connected dummy forward pass.

            Mark the model built first to prevent the dummy call from recursing
            into build(). The forward pass constructs the same sublayer state
            as a real call.
            """
            if self.built:
                return

            if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 1:
                first = input_shape[0]
                gene_shape = tf.TensorShape(
                    first if hasattr(first, "__len__") else input_shape
                )
            else:
                gene_shape = tf.TensorShape(input_shape)

            feature_dim = (
                int(gene_shape[-1]) if gene_shape.rank and gene_shape[-1] is not None
                else int(self.num_genes)
            )

            super().build(input_shape)

            n = 4
            dummy_genes = tf.zeros([n, feature_dim], dtype=tf.float32)
            # Fully-connect the n dummy nodes (k=n edges per query). A single
            # self-loop (k=1) would trigger Keras's "softmax over size-1 axis"
            # warning inside the transformer's MultiHeadAttention.
            sources = tf.repeat(tf.range(n, dtype=tf.int32), n)
            targets = tf.tile(tf.range(n, dtype=tf.int32), [n])
            dummy_edges = tf.stack([sources, targets], axis=0)
            _ = self([dummy_genes, dummy_edges], training=False, batch_size=n)

        def _build_encoder_layers(self):
            """Build shared encoder layers and dual variational heads."""

            # Create the front-end before graph attention and heads to match
            # positional checkpoint weight order.
            self.gene_frontend = None
            if self.use_gene_tokens:
                from . import encoder_frontend as _fe
                self.gene_frontend = _fe.GeneTokensFrontEnd(
                    embedding_table=self._gene_token_table,
                    col_to_row=self._gene_token_col_to_row,
                    k=self.gene_token_k,
                    token_dim=self.gene_token_dim,
                    output_dim=self.shared_dim,
                    num_heads=self.gene_token_heads,
                    dropout=0.0,
                    name="gene_tokens_frontend",
                )
            else:
                raise ValueError(
                    "This package loads gene-token models only; "
                    "checkpoint is not gene-token."
                )


            # Graph Attention Layer (Edge-Index Based)
            # Supports two types: GAT (additive) or Transformer (dot-product)
            if self.graph_attention_type == 'gat':
                # GAT: Additive attention with LeakyReLU (original GAT paper)
                self.gat_layer = EdgeIndexGAT(
                    units=self.shared_dim,
                    num_heads=self.num_attention_heads,
                    concat_heads=False,  # Average heads to maintain dimension
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.regularizer,
                    name="shared_gat_layer"
                )
            elif self.graph_attention_type == 'transformer':
                # Transformer: dot-product attention.
                self.gat_layer = EdgeIndexTransformer(
                    units=self.shared_dim,
                    num_heads=self.num_attention_heads,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    name="shared_transformer_layer"
                )

            else:
                raise ValueError(f"Unknown graph_attention_type: {self.graph_attention_type}. Must be 'gat' or 'transformer'.")


            # Final dropout and layer normalization before heads
            self.dropout_layer = tf.keras.layers.Dropout(
                rate=self.dropout_rate, 
                name="shared_dropout_final"
            )
            self.layer_norm = tf.keras.layers.LayerNormalization(
                epsilon=self.eps, 
                name="shared_layer_norm_final"
            )

            # Dual variational heads
            self.recon_head = VariationalHead(latent_dim=self.latent_dim, name='recon')
            self.class_head = VariationalHead(latent_dim=self.latent_dim, name='class')

        def encode_hidden(
            self, gene_values: tf.Tensor, training: Optional[bool] = None
        ) -> tf.Tensor:
            """Transform each node independently through the gene-token front-end.

            During inference, each row depends only on its gene values and the
            fixed species index. Callers can therefore cache hidden states of
            constant nodes, such as memory-bank anchors, across batches.

            Args:
                gene_values: Expression values ``[n_nodes, num_genes]``.
                training: Whether in training mode.

            Returns:
                Hidden node features ``[n_nodes, shared_dim]``.
            """
            if self.use_gene_tokens:
                # Append the per-batch checkpoint species index for the front-end.
                b = tf.shape(gene_values)[0]
                sp = tf.fill([b, 1], tf.cast(self.species_id, gene_values.dtype))
                x_aug = tf.concat([gene_values, sp], axis=1)          # [B, num_genes + 1]
                return self.gene_frontend(x_aug, training=training)   # [B, shared_dim]

            raise ValueError(
                "This package loads gene-token models only; "
                "checkpoint is not gene-token."
            )

        def encode_from_hidden(
            self,
            h: tf.Tensor,
            edge_index: tf.Tensor,
            edge_weights: Optional[tf.Tensor] = None,
            batch_size: Optional[int] = None,
            training: Optional[bool] = None,
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            """Run graph attention + dual heads on precomputed hidden features.

            Companion to :meth:`encode_hidden`. ``edge_weights`` is accepted for
            call-site symmetry but unused (the GAT layer takes only
            ``[h, edge_index]``).

            Args:
                h: Hidden node features ``[n_nodes, shared_dim]`` from
                   :meth:`encode_hidden`.
                edge_index: Graph edges ``[2, n_edges]``.
                edge_weights: Unused; kept for API symmetry with ``call``.
                batch_size: Optional row count to slice query outputs (Memory Bank).
                training: Whether in training mode.

            Returns:
                6-tuple of (z_recon, mu_recon, log_var_recon, z_class, mu_class,
                log_var_class).
            """
            # Apply GAT mechanism (edge-index based, avoids dense adjacency matrices)
            h_out = self.gat_layer(
                inputs=[h, edge_index],
                training=training,
                batch_size=batch_size
            )

            # Split into dual heads
            z_recon, mu_recon, log_var_recon = self.recon_head(h_out, training=training)
            z_class, mu_class, log_var_class = self.class_head(h_out, training=training)

            return z_recon, mu_recon, log_var_recon, z_class, mu_class, log_var_class

        def call(
            self,
            inputs: Union[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
            training: Optional[bool] = None,
            batch_size: Optional[int] = None,
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            """
            Forward pass of dual-head VGAE encoder.

            Args:
                inputs: Tuple of (gene_values, edge_index) or (gene_values, edge_index, edge_weights)
                training: Whether in training mode
                batch_size: Optional batch size for slicing output (if using Memory Bank)

            Returns:
                6-tuple of (z_recon, mu_recon, log_var_recon, z_class, mu_class, log_var_class)
            """
            # Parse inputs
            if len(inputs) == 2:
                gene_values, edge_index = inputs
                edge_weights = None
            elif len(inputs) == 3:
                gene_values, edge_index, edge_weights = inputs
            else:
                raise ValueError(f"Expected 2 or 3 inputs, got {len(inputs)}")

            # Separate node encoding from graph attention so callers can cache
            # hidden states for constant anchor nodes.
            h = self.encode_hidden(gene_values, training=training)
            return self.encode_from_hidden(
                h, edge_index, edge_weights, batch_size=batch_size, training=training
            )

def _match_gene_columns(adata_gene_ids, required_gene_ids):
    """Map required (model) gene IDs to source column indices in an AnnData.

    Plain string-equality matching (IDs may be integers, so both sides are
    stringified). Callers that only need the mapping -- e.g. the preflight
    memory guard's non-zero count -- can reuse it without materializing a
    reordered matrix.

    Args:
        adata_gene_ids: Iterable of the AnnData's gene IDs, in column order.
        required_gene_ids: Gene IDs the model expects, in model order.

    Returns:
        ``(source_indices, target_indices, n_found, n_missing)`` where
        ``source_indices[i]`` is the AnnData column holding the model gene placed
        at ``target_indices[i]``; both are ``np.intp`` arrays of length
        ``n_found``.
    """
    adata_gene_ids = [str(g) for g in adata_gene_ids]
    required_gene_ids = [str(g) for g in required_gene_ids]
    gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(adata_gene_ids)}
    src_list = []
    tgt_list = []
    for target_idx, gene_id in enumerate(required_gene_ids):
        idx = gene_id_to_idx.get(gene_id)
        if idx is not None:
            src_list.append(idx)
            tgt_list.append(target_idx)
    source_indices = np.array(src_list, dtype=np.intp)
    target_indices = np.array(tgt_list, dtype=np.intp)
    return source_indices, target_indices, len(src_list), len(required_gene_ids) - len(src_list)


# ============================================================================
# GRIT (Graph-Regularized Logit Refinement) Functions
# ============================================================================

def build_normalized_adjacency(edge_index: np.ndarray, 
                               edge_weights: np.ndarray, 
                               num_nodes: int) -> tf.SparseTensor:
    """
    Build a symmetrically normalized adjacency matrix (A_norm) from edge data.
    
    A_sym = maximum(A, A.T)
    A_norm = D^-0.5 * (A_sym + I) * D^-0.5
    D contains the row sums of A_sym + I.
    
    This normalized adjacency is used for label propagation in GRIT refinement.
    
    Args:
        edge_index: [2, num_edges] edge indices
        edge_weights: [num_edges] edge weights
        num_nodes: Number of nodes in the graph
        
    Returns:
        tf.SparseTensor: Normalized adjacency matrix
    """
    import scipy.sparse as sp
    
    # 1. Create sparse matrix A in COO format
    rows = edge_index[0]
    cols = edge_index[1]
    adj = sp.coo_matrix((edge_weights, (rows, cols)), shape=(num_nodes, num_nodes))
    
    # 2. Symmetrize by taking the maximum weight in each direction.
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # 3. Add self-loops (A_hat = A + I)
    adj_hat = adj + sp.eye(num_nodes)
    
    # 4. Create degree matrix D
    row_sum = np.array(adj_hat.sum(axis=1)).flatten()
    
    # Keep masked inverse degrees at zero with an initialized out array.
    # This prevents infinities or NaNs from propagating through normalization.
    d_inv_sqrt = np.power(row_sum, -0.5, out=np.zeros_like(row_sum),
                          where=row_sum > 0)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # 6. Compute A_norm = D^-0.5 * A_hat * D^-0.5
    adj_normalized = adj_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    
    # 7. Convert to tf.SparseTensor
    indices = np.vstack((adj_normalized.row, adj_normalized.col)).T
    values = adj_normalized.data.astype(np.float32)
    shape = adj_normalized.shape
    
    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


def apply_grit_refinement(
    initial_scores: Union[np.ndarray, tf.Tensor],
    adj_normalized_sparse: tf.SparseTensor,
    num_iterations: int = 3,
    alpha: float = 0.2,
    *,
    score_mode: str = "log_scores",
) -> Tuple[np.ndarray, Callable[[], None]]:
    """
    Refine initial scores using label propagation (GRIT).
    
    GRIT performs iterative label propagation on the k-NN graph to smooth predictions:
    Z_{k+1} = (1-α) * P_0 + α * A_norm * Z_k
    Each update is then row-normalized.
    
    Where:
    - P_0: Initial probabilities from the model
    - A_norm: Normalized adjacency matrix
    - α: Propagation weight (higher = more smoothing)
    - Z_k: Refined probabilities at iteration k
    
    The class dimension is processed in memory-bounded chunks: each iteration
    writes sparse-dense products into an output buffer, then normalizes rows.
    Chunk size is estimated from available system RAM; larger score buffers
    may use temporary memory-mapped files.
    
    Args:
        initial_scores: [n_cells, num_classes] initial logits/scores
        adj_normalized_sparse: Normalized adjacency matrix as tf.SparseTensor
        num_iterations: Number of propagation iterations (K)
        alpha: Propagation weight in [0, 1] (higher = more neighbor influence)
        score_mode: 'log_scores' applies softmax first; 'positive_scores'
            uses additive-epsilon row normalization.
        
    Returns:
        (refined_probs, cleanup), where refined_probs is a float32 NumPy array
        of shape [n_cells, num_classes], possibly memory-mapped. Consume or copy
        the result before calling cleanup, which closes temporary buffers.
    """
    initial_scores_array = np.asarray(initial_scores, dtype=np.float32)
    n_cells, n_classes = initial_scores_array.shape
    class_chunk_size = _get_cpu_class_chunk_size(
        n_cells,
        n_classes,
        working_buffers=3,
    )

    p_initial_buffer = _ScoreMatrixBuffer((n_cells, n_classes), np.float32)
    z_current_buffer = _ScoreMatrixBuffer(
        (n_cells, n_classes),
        np.float32,
        prefer_memmap=p_initial_buffer.is_memmap,
    )
    z_next_buffer = _ScoreMatrixBuffer(
        (n_cells, n_classes),
        np.float32,
        prefer_memmap=p_initial_buffer.is_memmap,
    )
    buffers = [p_initial_buffer, z_current_buffer, z_next_buffer]

    def _cleanup() -> None:
        for buffer in buffers:
            buffer.close()

    if score_mode not in {"log_scores", "positive_scores"}:
        raise ValueError(
            "score_mode must be 'log_scores' or 'positive_scores', got "
            f"{score_mode!r}"
        )

    if score_mode == "log_scores":
        row_max = np.full(n_cells, -np.inf, dtype=np.float32)
        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            block = np.asarray(initial_scores_array[:, start:end], dtype=np.float32)
            row_max = np.maximum(row_max, np.max(block, axis=1))

        denom = np.zeros(n_cells, dtype=np.float64)
        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            block = np.asarray(initial_scores_array[:, start:end], dtype=np.float32)
            probs = np.exp(block - row_max[:, None]).astype(
                np.float32, copy=False
            )
            denom += np.sum(probs, axis=1, dtype=np.float64)

        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            block = np.asarray(initial_scores_array[:, start:end], dtype=np.float32)
            probs = np.exp(block - row_max[:, None]).astype(
                np.float32, copy=False
            )
            probs /= denom[:, None]
            p_initial_buffer.array[:, start:end] = probs.astype(np.float32, copy=False)
            z_current_buffer.array[:, start:end] = probs.astype(np.float32, copy=False)
    else:
        eps = 1e-15
        denom = np.full(n_cells, eps * n_classes, dtype=np.float64)
        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            block = np.asarray(initial_scores_array[:, start:end], dtype=np.float32)
            denom += np.sum(block, axis=1, dtype=np.float64)

        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            block = np.asarray(initial_scores_array[:, start:end], dtype=np.float32)
            probs = (block + eps) / denom[:, None]
            p_initial_buffer.array[:, start:end] = probs.astype(np.float32, copy=False)
            z_current_buffer.array[:, start:end] = probs.astype(np.float32, copy=False)

    with tf.device("/CPU:0"):
        for _ in range(num_iterations):
            row_sums = np.zeros(n_cells, dtype=np.float64)
            for start in range(0, n_classes, class_chunk_size):
                end = min(start + class_chunk_size, n_classes)
                current_block = np.asarray(
                    z_current_buffer.array[:, start:end],
                    dtype=np.float32,
                )
                chunk_prop = tf.sparse.sparse_dense_matmul(
                    adj_normalized_sparse,
                    tf.convert_to_tensor(current_block, dtype=tf.float32),
                ).numpy()
                mixed = (
                    (1.0 - alpha)
                    * np.asarray(p_initial_buffer.array[:, start:end], dtype=np.float32)
                    + alpha * chunk_prop
                )
                z_next_buffer.array[:, start:end] = mixed.astype(np.float32, copy=False)
                row_sums += np.sum(mixed, axis=1, dtype=np.float64)

            row_sums += 1e-10
            for start in range(0, n_classes, class_chunk_size):
                end = min(start + class_chunk_size, n_classes)
                z_next_buffer.array[:, start:end] /= row_sums[:, None]

            z_current_buffer, z_next_buffer = z_next_buffer, z_current_buffer
            buffers[1], buffers[2] = z_current_buffer, z_next_buffer

    return z_current_buffer.array, _cleanup

def rollup_rare_annotations(
    predictor: HECTOR,
    predictions: pd.DataFrame,
    min_cells_pct: float = 1.0,
    max_cells: int = 20,
    rare_label: str = 'Rare Cells',
) -> pd.DataFrame:
    """Bucket rare cell-type annotations below a population threshold.

    Thin wrapper around :meth:`HECTOR._run_rollup` for use in
    custom workflows that need the bucketed DataFrame without writing to
    AnnData.

    Args:
        predictor: A loaded :class:`HECTOR` instance.
        predictions: DataFrame returned by ``predictor.predict()``.  Must
            contain a ``top_1_prediction`` column.
        min_cells_pct: Minimum percentage of total cells a cell type must
            reach to avoid being bucketed.  Must be in ``[0.0, 100.0]``.
        max_cells: Hard cap on the computed threshold.
        rare_label: Label assigned to cell types below the threshold.

    Returns:
        A copy of *predictions* with two additional columns:
        ``bucketed_annotation`` (the possibly relabeled label) and
        ``is_bucketed`` (``True`` when the cell was moved into the rare
        bucket).

    Raises:
        ValueError: If *min_cells_pct* is outside ``[0.0, 100.0]``.
    """
    if min_cells_pct < 0.0 or min_cells_pct > 100.0:
        raise ValueError(
            f"min_cells_pct must be between 0.0 and 100.0, got {min_cells_pct}"
        )

    bucketed, is_bucketed = predictor._run_rollup(
        predictions['top_1_prediction'], min_cells_pct, max_cells, rare_label
    )

    result = predictions.copy()
    result['bucketed_annotation'] = bucketed.values
    result['is_bucketed'] = is_bucketed.values
    return result


# ============================================================================
# Convenience Functions
# ============================================================================

def predict_from_anndata(
    adata,
    model_path: Union[str, Path, None] = None,
    top_k: int = 5,
    use_full_ontology: bool = True,
    export_score_matrix: bool = False,
    auto_download: bool = True,
    **kwargs
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function for quick predictions.

    Args:
        adata: AnnData object
        model_path: Path to a checkpoint (.h5). None selects the registered
            HECTOR Pro model, downloading it when auto_download is True.
        top_k: Number of top predictions to return
        use_full_ontology: Whether to use zero-shot mode
        export_score_matrix: Whether to also return the full score matrix
        auto_download: Whether to download a missing registered model automatically
        **kwargs: Additional arguments passed to InferenceConfig

    Returns:
        Predictions DataFrame, or (predictions, score_matrix) when
        export_score_matrix is True.

    Example:
        predictions = predict_from_anndata(
            adata,
            model_path="model.h5",
            top_k=5
        )
    """
    config = InferenceConfig(
        top_k=top_k,
        use_full_ontology=use_full_ontology,
        **kwargs
    )

    predictor = HECTOR(
        model_path,
        config=config,
        auto_download=auto_download,
    )
    return predictor.predict(
        adata,
        export_score_matrix=export_score_matrix,
    )

def _discover_cellranger_candidates(
    data_path: Union[str, Path]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Recursively discover Cell Ranger MTX directories and H5 files."""
    search_path = Path(data_path).expanduser()
    search_root = search_path if search_path.is_dir() else search_path.parent

    candidates: List[Dict[str, Any]] = []
    diagnostics: List[str] = []

    def infer_sample_name(candidate_path: Path) -> Tuple[str, bool]:
        path_parts = list(candidate_path.parts)
        path_parts_lower = [part.lower() for part in path_parts]

        if "per_sample_outs" in path_parts_lower:
            sample_index = path_parts_lower.index("per_sample_outs") + 1
            if sample_index < len(path_parts):
                return path_parts[sample_index], True

        if "outs" in path_parts_lower:
            outs_index = path_parts_lower.index("outs")
            if outs_index > 0:
                return path_parts[outs_index - 1], False

        generic_names = {
            "count",
            "multi",
            "outs",
            "filtered_feature_bc_matrix",
            "raw_feature_bc_matrix",
            "filtered_gene_bc_matrices",
            "raw_gene_bc_matrices",
        }

        search_start = candidate_path.parent if candidate_path.suffix else candidate_path
        for ancestor in [search_start, *search_start.parents]:
            ancestor_name = ancestor.name
            ancestor_name_lower = ancestor_name.lower()
            if not ancestor_name:
                continue
            if ancestor_name_lower in generic_names:
                continue
            if ancestor_name_lower.endswith("_filtered_feature_bc_matrix"):
                continue
            if ancestor_name_lower.endswith("_raw_feature_bc_matrix"):
                continue
            return ancestor_name, False

        return "sample", False

    def classify_filtered_state(candidate_path: Path) -> str:
        path_tokens = [
            token for token in re.split(r"[^a-z0-9]+", str(candidate_path).lower()) if token
        ]
        if "filtered" in path_tokens:
            return "filtered"
        if "raw" in path_tokens:
            return "raw"
        return "unknown"

    def add_candidate(candidate_path: Path, source_type: str) -> None:
        sample_name, is_sample_level = infer_sample_name(candidate_path)
        try:
            depth = len(candidate_path.relative_to(search_root).parts)
        except ValueError:
            depth = len(candidate_path.parts)

        filtered_state = classify_filtered_state(candidate_path)
        candidates.append(
            {
                "path": str(candidate_path),
                "source_type": source_type,
                "filtered_state": filtered_state,
                "sample_name": sample_name,
                "is_sample_level": is_sample_level,
                "depth": depth,
            }
        )
        diagnostics.append(
            f"✅ {source_type.upper()} candidate "
            f"[{filtered_state}, sample='{sample_name}'] -> {candidate_path}"
        )

    if not search_path.exists():
        diagnostics.append(f"❌ Path does not exist: {search_path}")
        return candidates, diagnostics

    if search_path.is_file():
        lower_name = search_path.name.lower()
        if "feature_bc_matrix" in lower_name and lower_name.endswith(".h5"):
            add_candidate(search_path, "h5")
        else:
            diagnostics.append(
                f"⚠️ Ignored file without supported 10x pattern -> {search_path}"
            )
        return candidates, diagnostics

    for root, _, files in os.walk(search_path):
        root_path = Path(root)
        lower_files = {file_name.lower() for file_name in files}

        has_matrix = any(
            name in lower_files for name in ("matrix.mtx", "matrix.mtx.gz")
        )
        has_features = any(
            name in lower_files
            for name in (
                "features.tsv",
                "features.tsv.gz",
                "genes.tsv",
                "genes.tsv.gz",
            )
        )
        has_barcodes = any(
            name in lower_files for name in ("barcodes.tsv", "barcodes.tsv.gz")
        )

        if has_matrix or has_features or has_barcodes:
            if has_matrix and has_features and has_barcodes:
                add_candidate(root_path, "mtx")
            else:
                missing_parts = []
                if not has_matrix:
                    missing_parts.append("matrix.mtx(.gz)")
                if not has_features:
                    missing_parts.append("features.tsv(.gz) or genes.tsv(.gz)")
                if not has_barcodes:
                    missing_parts.append("barcodes.tsv(.gz)")
                diagnostics.append(
                    f"⚠️ Partial MTX directory missing {', '.join(missing_parts)} -> {root_path}"
                )

        for file_name in sorted(files):
            lower_name = file_name.lower()
            if "feature_bc_matrix" in lower_name and lower_name.endswith(".h5"):
                add_candidate(root_path / file_name, "h5")

    return candidates, diagnostics


def _format_loaded_cellranger_adata(adata, sample_name: str):
    """Standardize AnnData metadata from 10x MTX and H5 readers."""
    adata.obs_names = pd.Index(adata.obs_names.astype(str))
    adata.var_names = pd.Index(adata.var_names.astype(str))

    feature_id_column = None
    for column_name in ("gene_ids", "feature_ids", "id", "feature_id"):
        if column_name in adata.var.columns:
            feature_id_column = column_name
            break

    original_var_names = adata.var_names.astype(str).to_numpy(copy=True)
    if feature_id_column is not None:
        feature_id_values = adata.var[feature_id_column].astype(str).to_numpy(copy=True)
    else:
        feature_id_values = original_var_names

    gene_symbol_column = None
    for column_name in ("gene_symbols", "gene_symbol", "feature_name", "gene_name", "name"):
        if column_name in adata.var.columns:
            gene_symbol_column = column_name
            break

    if gene_symbol_column is not None:
        gene_symbol_values = adata.var[gene_symbol_column].astype(str).to_numpy(copy=True)
    else:
        gene_symbol_values = original_var_names

    adata.var_names = pd.Index(feature_id_values.astype(str))

    adata.var_names_make_unique()
    adata.var["feature_id"] = adata.var_names.astype(str)
    adata.var["gene_symbols"] = gene_symbol_values

    for column_name in ("gene_ids", "feature_ids", "id"):
        if column_name in adata.var.columns:
            adata.var = adata.var.drop(columns=[column_name])

    barcode_values = adata.obs_names.astype(str)
    adata.obs["sample"] = sample_name
    adata.obs["barcode"] = barcode_values
    adata.obs_names = pd.Index(
        [f"{sample_name}_{barcode}" for barcode in barcode_values]
    )

    return adata


def _load_cellranger_candidate(candidate: Dict[str, Any]):
    """Load one Cell Ranger candidate using Scanpy's 10x readers."""
    import scanpy as sc

    if candidate["source_type"] == "mtx":
        adata = sc.read_10x_mtx(
            candidate["path"],
            var_names="gene_ids",
            make_unique=False,
            gex_only=False,
        )
    elif candidate["source_type"] == "h5":
        adata = sc.read_10x_h5(candidate["path"], gex_only=False)
    else:
        raise ValueError(f"Unsupported Cell Ranger candidate type: {candidate['source_type']}")

    return _format_loaded_cellranger_adata(adata, candidate["sample_name"])


def _restore_concatenated_cellranger_var_metadata(
    adata,
    source_adatas: List[anndata.AnnData],
):
    """Restore key var metadata after outer-join concatenation."""
    merged_feature_ids = pd.Series(index=adata.var_names, dtype=object)
    merged_gene_symbols = pd.Series(index=adata.var_names, dtype=object)

    for source_adata in source_adatas:
        source_var = source_adata.var.reindex(source_adata.var_names)
        if "feature_id" in source_var.columns:
            merged_feature_ids = merged_feature_ids.combine_first(
                source_var["feature_id"].reindex(adata.var_names)
            )
        if "gene_symbols" in source_var.columns:
            merged_gene_symbols = merged_gene_symbols.combine_first(
                source_var["gene_symbols"].reindex(adata.var_names)
            )

    merged_gene_symbols = merged_gene_symbols.fillna(pd.Series(adata.var_names, index=adata.var_names))
    merged_feature_ids = merged_feature_ids.fillna(merged_gene_symbols)

    adata.var["gene_symbols"] = merged_gene_symbols.astype(str).to_numpy()
    adata.var["feature_id"] = adata.var_names.astype(str)
    return adata


def diagnose_cellranger_directory(path, candidate_messages: Optional[List[str]] = None):
    """
    Diagnostic helper to list directory contents when files are not found.
    Uses simplified recursion to show relevant structure.
    """
    import os
    if not os.path.exists(path):
        print(f"  ❌ Path '{path}' does not exist.")
        return

    if os.path.isfile(path):
        print(f"  Existing file: '{path}'")
        print(f"    📄 {os.path.basename(path)}")
        if candidate_messages:
            print("  Candidate scan:")
            for message in candidate_messages[:20]:
                print(f"    {message}")
            if len(candidate_messages) > 20:
                print(f"    ... and {len(candidate_messages) - 20} more entries")
        return

    print(f"  Existing contents of '{path}':")
    try:
        # Walk just to depth 2 to show relevant subfolders
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            if level > 1:
                continue

            indent = "  " * (level + 1)
            if root != path:
                print(f"{indent}📁 {os.path.basename(root)}/")

            subindent = "  " * (level + 2)
            for f in sorted(files):
                print(f"{subindent}📄 {f}")

    except Exception as e:
        print(f"  ❌ Error listing directory: {e}")

    if candidate_messages:
        print("  Candidate scan:")
        for message in candidate_messages[:20]:
            print(f"    {message}")
        if len(candidate_messages) > 20:
            print(f"    ... and {len(candidate_messages) - 20} more entries")


def load_cellranger_data(data_path):
    """
    Loads Cell Ranger output with recursive discovery for conventional and multiplex layouts.

    Args:
        data_path (str): Path to a Cell Ranger output root, matrix directory, or H5 file.
    Returns:
        anndata.AnnData: The raw count matrix.
    """
    candidates, candidate_messages = _discover_cellranger_candidates(data_path)

    def candidate_priority(candidate: Dict[str, Any]) -> Tuple[int, int, int, int]:
        filtered_priority = {"filtered": 0, "unknown": 1, "raw": 2}
        source_priority = {"mtx": 0, "h5": 1}
        return (
            0 if candidate["is_sample_level"] else 1,
            filtered_priority.get(candidate["filtered_state"], 1),
            source_priority.get(candidate["source_type"], 1),
            candidate["depth"],
        )

    selected_candidates: List[Dict[str, Any]] = []
    ambiguous_candidates: Optional[List[Dict[str, Any]]] = None

    sample_candidates = [candidate for candidate in candidates if candidate["is_sample_level"]]
    if sample_candidates:
        for sample_name in sorted({candidate["sample_name"] for candidate in sample_candidates}):
            sample_options = [
                candidate
                for candidate in sample_candidates
                if candidate["sample_name"] == sample_name
            ]
            sample_options.sort(
                key=lambda candidate: (candidate_priority(candidate), candidate["path"])
            )
            selected_candidates.append(sample_options[0])

        selected_candidates.sort(
            key=lambda candidate: (
                candidate["sample_name"],
                candidate_priority(candidate),
                candidate["path"],
            )
        )
    elif candidates:
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (candidate_priority(candidate), candidate["path"]),
        )
        top_priority = candidate_priority(ranked_candidates[0])
        tied_candidates = [
            candidate
            for candidate in ranked_candidates
            if candidate_priority(candidate) == top_priority
        ]
        if len(tied_candidates) > 1:
            ambiguous_candidates = tied_candidates
        else:
            selected_candidates = [ranked_candidates[0]]

    if ambiguous_candidates:
        print(f"\n❌ Found multiple equally plausible Cell Ranger outputs in {data_path}")
        ambiguity_messages = candidate_messages + [
            f"⚠️ Ambiguous top-ranked candidate -> {candidate['path']}"
            for candidate in ambiguous_candidates
        ]
        diagnose_cellranger_directory(data_path, ambiguity_messages)
        raise ValueError(
            "Ambiguous Cell Ranger output files found. "
            "Please pass a more specific sample or matrix directory."
        )

    if not selected_candidates:
        print(f"\n❌ Could not find valid Cell Ranger output in {data_path}")
        diagnose_cellranger_directory(data_path, candidate_messages)
        raise FileNotFoundError("Missing or incomplete Cell Ranger output files.")

    try:
        if len(selected_candidates) > 1:
            print(
                f"  ✅ Found {len(selected_candidates)} sample-level Cell Ranger outputs; loading and concatenating..."
            )
            adata_list = []
            for candidate in selected_candidates:
                print(f"  ⬇️  Loading sample '{candidate['sample_name']}' from: {candidate['path']}")
                sample_adata = _load_cellranger_candidate(candidate)
                adata_list.append(sample_adata)

            adata = anndata.concat(
                adata_list,
                join="outer",
                merge="same",
            )
            adata = _restore_concatenated_cellranger_var_metadata(adata, adata_list)
        else:
            candidate = selected_candidates[0]
            print(f"  ✅ Found data in: {candidate['path']}")
            adata = _load_cellranger_candidate(candidate)

        print(f"  ✅ Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
        return adata

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        diagnose_cellranger_directory(data_path, candidate_messages)
        raise

# ---------------------------------------------------------------------------
# HECTOR model loaders
# ---------------------------------------------------------------------------
def _load_models_from_hdf5(self):
    """Load model components from an HDF5 checkpoint."""
    import json
    
    with h5py.File(self.model_path, 'r') as f:
        # Load model config
        model_config = json.loads(f['config'].attrs['model_config'])
        self._log(f"  ☑️  Model config loaded successfully")
        
        # Helper to load string lists from HDF5 group
        def _load_string_list(h5_group, key):
            if key in h5_group:
                return [s.decode('utf-8') if isinstance(s, bytes) else s for s in h5_group[key][:]]
            return []

        # Load ontology data
        ontology_grp = f['ontology']
        ontology_data = {
            'ontology_adj': ontology_grp['ontology_adj'][:] if 'ontology_adj' in ontology_grp else None,
            'semantic_embeddings': json.loads(ontology_grp.attrs['semantic_embeddings']) if 'semantic_embeddings' in ontology_grp.attrs else None,
            'classes': _load_string_list(ontology_grp, 'classes'),
            'structural_dim': ontology_grp.attrs.get('structural_dim', 128),
            'ontology_prior': ontology_grp['ontology_prior'][:] if 'ontology_prior' in ontology_grp else None,
            'full_ontology_adj': ontology_grp['full_ontology_adj'][:] if 'full_ontology_adj' in ontology_grp else None,
            'full_semantic_embeddings': json.loads(ontology_grp.attrs['full_semantic_embeddings']) if 'full_semantic_embeddings' in ontology_grp.attrs else None,
            'full_classes': _load_string_list(ontology_grp, 'full_classes'),
            'full_class_names': _load_string_list(ontology_grp, 'full_class_names'),
            # IS_A DAG only (no k-NN edges). adj[child, parent] = 1. Use for hierarchy/visualization.
            'pure_ontology_adj': ontology_grp['pure_ontology_adj'][:] if 'pure_ontology_adj' in ontology_grp else None,
            'full_pure_ontology_adj': ontology_grp['full_pure_ontology_adj'][:] if 'full_pure_ontology_adj' in ontology_grp else None,
            # Level arrays for hierarchical level embeddings
            'level_array': ontology_grp['level_array'][:] if 'level_array' in ontology_grp else None,
            'full_level_array': ontology_grp['full_level_array'][:] if 'full_level_array' in ontology_grp else None,
        }
        self._log("  ☑️  Ontology data loaded successfully")
        
        # Log level embedding status
        use_level_emb = model_config.get('use_level_embeddings', True)
        has_level_arrays = (
            ontology_data.get('level_array') is not None
            and ontology_data.get('full_level_array') is not None
        )
        if not use_level_emb:
            self._log("  ℹ️  Level embeddings disabled in checkpoint config")
        elif not has_level_arrays:
            self._log("  ⚠️  GAT level arrays not found in checkpoint (level embeddings disabled)")
        
        self._log("  ⏳ Loading model weights...")
        
        # Helper function to load weights from HDF5
        def load_weights_from_h5(h5_group, prefix='weight'):
            """Load list of weight arrays from HDF5 group."""
            weights = []
            i = 0
            while f'{prefix}_{i}' in h5_group:
                dataset = h5_group[f'{prefix}_{i}']
                if dataset.shape == ():
                    weights.append(dataset[()])
                else:
                    weights.append(dataset[:])
                i += 1
            return weights
        
        # Load weights
        weights_grp = f['weights']
        
        # Encoder weights
        encoder_grp = weights_grp['vgae_encoder']
        is_dual_head = encoder_grp.attrs.get('is_dual_head', False)

        if not is_dual_head or not model_config['encoder_config']['use_gene_tokens']:
            raise ValueError('Only dual-head gene-token checkpoints are supported.')
        shared_weights = load_weights_from_h5(encoder_grp['shared_encoder'])
        recon_head_weights = load_weights_from_h5(encoder_grp['recon_head'])
        class_head_weights = load_weights_from_h5(encoder_grp['class_head'])
        vgae_encoder_weights = shared_weights + recon_head_weights + class_head_weights
        # These named datasets are hard links to shared weights in current exports.
        assets = encoder_grp['gene_token_assets']
        gene_token_col_to_row = shared_weights[0]
        gene_token_table = shared_weights[1]
        if (not np.array_equal(gene_token_col_to_row, assets['col_to_row'][:])
                or gene_token_table.shape != assets['protein_embeddings'].shape):
            raise ValueError('Named gene-token assets disagree with encoder weights.')
        use_gene_tokens = True
        
        # GAT weights
        celltype_gat_weights = load_weights_from_h5(weights_grp['celltype_gat'])
    
    # Reconstruct models from checkpoint configuration.
    config = dict(model_config)
    config.update(model_config['encoder_config'])
    
    # Reconstruct VGAE Encoder
    
    # Handle hidden_dims
    hidden_dims_from_checkpoint = config.get('hidden_dims', None)
    if hidden_dims_from_checkpoint is not None and len(hidden_dims_from_checkpoint) == 0:
        hidden_dims_from_checkpoint = None
    
    vgae_encoder = DualHeadVGAEEncoder(
        num_genes=config.get('num_genes', config['input_dim']),
        latent_dim=config['latent_dim'],
        hidden_dims=hidden_dims_from_checkpoint,
        shared_dim=config.get('shared_dim', 1024),
        num_attention_heads=config.get('num_attention_heads', 4),
        dropout_rate=config.get('dropout_rate', 0.3),
        activation=config.get('activation', 'relu'),
        use_bias=config.get('use_bias', True),
        kernel_regularizer=config.get('kernel_regularizer', None),
        graph_attention_type=config.get('graph_attention_type', 'transformer'),
        use_gene_tokens=use_gene_tokens,
        gene_token_table=gene_token_table,
        gene_token_col_to_row=gene_token_col_to_row,
        gene_token_k=config.get('gene_token_k', 2048),
        gene_token_dim=config.get('gene_token_dim', 1024),
        gene_token_heads=config.get('gene_token_heads', 8),
        numerical_stability_eps=config['numerical_stability_eps'],
    )
    
    # Reconstruct CellType GAT
    
    # Set GAT architecture parameters from checkpoint
    # Architecture parameters must match the saved weights.
    gat_num_layers = config.get('gat_num_layers', None)
    if gat_num_layers is None:
        raise ValueError(
            "Checkpoint does not contain 'gat_num_layers' in model_config. "
            "This checkpoint was created with an older version of the code. "
            "Please retrain your model to generate a compatible checkpoint."
        )
    Config.GAT_NUM_LAYERS = gat_num_layers
    
    # Load the number of GAT heads, defaulting to 4.
    gat_num_heads = config.get('gat_num_heads', 4)
    Config.GAT_NUM_HEADS = gat_num_heads        
    # Load dropout parameters with defaults for checkpoints lacking them.
    gat_attention_dropout = config.get('gat_attention_dropout', 0.1)
    Config.GAT_ATTENTION_DROPOUT = gat_attention_dropout
    
    gat_dropout = config.get('gat_dropout', 0.1)
    Config.GAT_DROPOUT = gat_dropout
    
    # Load level embedding configuration
    use_level_embeddings = config.get('use_level_embeddings', True)
    level_embedding_dim = config.get('level_embedding_dim', 32)
    hpl_level_embedding_dim = config.get('hpl_level_embedding_dim', 32)
    
    Config.USE_LEVEL_EMBEDDINGS = use_level_embeddings
    Config.LEVEL_EMBEDDING_DIM = level_embedding_dim
    Config.HPL_LEVEL_EMBEDDING_DIM = hpl_level_embedding_dim
    Config.GAT_ACTIVATION = config['gat_activation']
    
    celltype_gat = CellTypeEmbedder(
        ontology_adj=ontology_data['ontology_adj'],
        semantic_embeddings=ontology_data['semantic_embeddings'],
        classes=ontology_data['classes'],
        structural_dim=ontology_data['structural_dim'],
        full_ontology_adj=ontology_data.get('full_ontology_adj'),
        full_semantic_embeddings=ontology_data.get('full_semantic_embeddings'),
        full_classes=ontology_data.get('full_classes', []),
        level_array=ontology_data.get('level_array'),
        full_level_array=ontology_data.get('full_level_array'),
        
        # Pass PPR-Based Prior configuration
        prior_method=config.get('prior_method', 'ppr'),
        ppr_alpha=config.get('ppr_alpha', 0.15),
        prior_matrix=ontology_data.get('ontology_prior')
    )
    celltype_gat.zero_shot_mode = bool(config['gat_zero_shot_mode'])
    
    # Build models with dummy data
    # Use 2+ nodes with proper k-NN edges to avoid softmax axis=1 warning
    # (softmax over neighbors requires >1 neighbor for meaningful computation)
    dummy_batch_size = 2
    dummy_k = 2  # k neighbors per node
    dummy_x = tf.zeros((dummy_batch_size, config['input_dim']), dtype=tf.float32)
    # Create edge_index: each node connects to all nodes (including self)
    # Shape: [2, batch_size * k] where row 0 = source nodes, row 1 = target nodes
    dummy_edge_index = tf.constant([
        [0, 0, 1, 1],  # source nodes (repeated k times each)
        [0, 1, 0, 1]   # target nodes (neighbors)
    ], dtype=tf.int32)
    dummy_edge_weights = tf.ones((dummy_batch_size * dummy_k,), dtype=tf.float32)
    
    # Suppress Keras build() warning for DualHeadVGAEEncoder
    # The encoder uses lazy layer building which triggers this benign warning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*build.*was called on layer.*')
        _ = vgae_encoder([dummy_x, dummy_edge_index, dummy_edge_weights], training=False, batch_size=dummy_batch_size)
    
    dummy_z = tf.zeros((1, config['latent_dim']), dtype=tf.float32)
    
    _ = celltype_gat(dummy_z, training=False)
    
    # Check weight counts before assignment to report architecture mismatches.
    n_model = len(vgae_encoder.get_weights())
    n_ckpt = len(vgae_encoder_weights)
    if n_model != n_ckpt:
        raise ValueError(
            f"VGAE encoder weight-count mismatch: reconstructed model has {n_model} "
            f"weight tensors but the checkpoint has {n_ckpt} (gene_token mode="
            f"{use_gene_tokens}). The encoder architecture does not match the checkpoint."
        )
    vgae_encoder.set_weights(vgae_encoder_weights)

    try:
        celltype_gat.set_weights(celltype_gat_weights)
        self._log("  ☑️  Model weights loaded successfully")
    except ValueError as e:
        expected_weights = len(celltype_gat.get_weights())
        actual_weights = len(celltype_gat_weights)
        # True architecture mismatch — provide detailed diagnostics
        error_msg = str(e)
        self._log(f"  ❌ Failed to load CellTypeEmbedder weights: {error_msg}")
        self._log(f"  ❌ Weight count mismatch: checkpoint has {actual_weights}, model expects {expected_weights}")
        self._log(f"  ❌ Current Config: GAT_NUM_HEADS={Config.GAT_NUM_HEADS}, GAT_NUM_LAYERS={Config.GAT_NUM_LAYERS}")
        self._log(f"  ❌ Checkpoint cfg 'gat_num_heads': {config.get('gat_num_heads', 'NOT FOUND')}")
        self._log(f"  ❌ Checkpoint cfg 'gat_num_layers': {config.get('gat_num_layers', 'NOT FOUND')}")
        raise ValueError(
                f"CellTypeEmbedder weight mismatch. Checkpoint has {actual_weights} weights, "
                f"model expects {expected_weights}. The checkpoint's 'gat_num_heads' config value "
                f"({config.get('gat_num_heads', 'NOT FOUND')}) may not match the actual saved architecture. "
            ) from e
    
    # IS_A DAG only (no k-NN). adj[child, parent] = 1. Use for hierarchy/visualization.
    # For the GAT input graph (IS_A + k-NN), see celltype_gat.full_ontology_adj_np.
    self.pure_ontology_adj = ontology_data.get('pure_ontology_adj')
    self.full_pure_ontology_adj = ontology_data.get('full_pure_ontology_adj')
    
    # Store cell type names and create ID-to-name mapping
    self.full_class_names = ontology_data.get('full_class_names', [])
    self.id_to_name_map = {}
    
    if self.full_class_names and hasattr(celltype_gat, 'full_classes') and celltype_gat.full_classes:
        full_classes = celltype_gat.full_classes
        
        # Validate length alignment
        if len(self.full_class_names) == len(full_classes):
            self.id_to_name_map = dict(zip(full_classes, self.full_class_names))
        else:
            self._log(f"  ⚠️  Name count mismatch: {len(self.full_class_names)} names vs {len(full_classes)} IDs")
            self._log(f"  ⚠️  Cell type names will not be available")
    else:
        if not self.full_class_names:
            self._log(f"  ⚠️  Cell type names not available in checkpoint (backward compatibility mode)")
    
    return vgae_encoder, celltype_gat
def _load_hpl_head(self):
    """Load HPL head from checkpoint if available."""
    try:
        with h5py.File(self.model_path, 'r') as f:
            # Check if aux_classifier exists
            if 'weights/aux_classifier' not in f:
                self._log("  ⚠️  No HPL head in checkpoint (will use GAT-only predictions)")
                return None
            
            # Load aux_classifier config
            config_str = f['config'].attrs.get('model_config')
            if config_str is None:
                self._log("  ⚠️  No model_config in checkpoint")
                return None
            
            import json
            config = json.loads(config_str)
            aux_config = config.get('aux_classifier_config')
            
            if aux_config is None:
                self._log("  ⚠️  No aux_classifier_config in checkpoint")
                return None
            
            # Get HPL parameters
            num_total_nodes = aux_config['num_total_nodes']
            num_seen_classes = aux_config['num_seen_classes']
            embedding_dim = aux_config['embedding_dim']
            scale = aux_config.get('scale', 20.0)
            
            # Load ancestry matrix
            if 'config/ancestry_matrix' in f:
                ancestry_matrix = f['config/ancestry_matrix'][:]
                ancestry_matrix_tensor = tf.constant(ancestry_matrix, dtype=tf.float32)
            else:
                self._log("  ⚠️  No ancestry_matrix (creating dummy)")
                import numpy as np
                ancestry_matrix = np.zeros((num_seen_classes, num_total_nodes), dtype=np.float32)
                for i in range(min(num_seen_classes, num_total_nodes)):
                    ancestry_matrix[i, i] = 1.0
                ancestry_matrix_tensor = tf.constant(ancestry_matrix, dtype=tf.float32)
            
            # Load node_levels for HPL level embeddings (backward compatible)
            node_levels = None
            if 'config/hpl_node_levels' in f:
                node_levels = f['config/hpl_node_levels'][:]
            elif not config.get('use_level_embeddings', True):
                # Level embeddings disabled in config, no warning needed
                pass
            else:
                self._log("  ⚠️  HPL node_levels not found in checkpoint (level embeddings disabled)")
            
            # Create HPL head
            aux_classifier = HPLHead(
                num_total_nodes=num_total_nodes,
                embedding_dim=embedding_dim,
                ancestry_matrix=ancestry_matrix_tensor,
                scale=scale,
                node_levels=node_levels
            )
            
            # Build model with dummy data
            dummy_embeddings = tf.zeros((1, embedding_dim), dtype=tf.float32)
            _ = aux_classifier(dummy_embeddings, training=False)
            
            # Load weights using the correct key pattern (weight_0, weight_1, etc.)
            def load_weights_from_h5(h5_group, prefix='weight'):
                """Load list of weight arrays from HDF5 group."""
                weights = []
                i = 0
                while f'{prefix}_{i}' in h5_group:
                    dataset = h5_group[f'{prefix}_{i}']
                    if dataset.shape == ():
                        weights.append(dataset[()])
                    else:
                        weights.append(dataset[:])
                    i += 1
                return weights
            
            aux_weights = load_weights_from_h5(f['weights/aux_classifier'], prefix='weight')
            
            if not aux_weights:
                self._log(f"  ⚠️  No weights found in aux_classifier group")
                return None
            
            aux_classifier.set_weights(aux_weights)
            
            # Verify what was loaded
            if len(aux_weights) >= 3:
                 pass
            # Checkpoint includes scale and margin weights.
            elif len(aux_weights) == 1:
            # Checkpoint contains only node_parts.
                # Set reasonable fallback value for scale (margin will be reset below)
                aux_classifier.scale.assign(12.26)  # Typical trained value

            else:
                self._log(f"  ⚠️  Unexpected number of weights: {len(aux_weights)}")
            
            # Reset the training-only angular margin for cosine-similarity inference.
            original_margin = float(aux_classifier.margin.numpy())
            aux_classifier.margin.assign(0.0)
            if original_margin != 0.0:
                pass
            
            return aux_classifier
            
    except Exception as e:
        self._log(f"  ⚠️  Failed to load HPL head: {e}")
        import traceback
        traceback.print_exc()
        return None
def _load_memory_bank(self) -> Optional[tf.Tensor]:
    """
    Load Memory Bank (anchor cells) from the checkpoint.

    Return the valid prefix of the saved anchor pool without confidence filtering.

    Returns:
        anchor_features: [num_anchors, num_backbone_genes] anchor cell features,
                       or None if Memory Bank is not present in the checkpoint.
    """
    try:
        with h5py.File(self.model_path, 'r') as f:
            if 'memory_bank' not in f:
                self._log("  ℹ️  No Memory Bank in checkpoint (will use batch-only graphs)")
                return None

            mem_grp = f['memory_bank']

            mem_size = mem_grp['mem_weight_1'][()]
            mem_features = mem_grp['mem_weight_2'][:]

            current_size = int(mem_size)
            raw_anchor_features = mem_features[:current_size]

            self._log(f"  📂 Loaded Memory Bank: {current_size:,} anchor cells")
            return tf.constant(raw_anchor_features, dtype=tf.float32)

    except Exception as e:
        self._log(f"  ⚠️  Failed to load Memory Bank: {e}")
        import traceback
        traceback.print_exc()
        return None
def _load_rotation_matrix(self) -> Optional[np.ndarray]:
    """
    Load Procrustes rotation matrix and centroid means from checkpoint.
    
    The rotation matrix R aligns GAT embeddings to HPL embeddings.
    
    For enhanced Procrustes (with mean centering), the alignment is:
        aligned_gat = (gat_embeddings - gat_mean) @ R + hpl_mean
    
    For standard Procrustes (no centering):
        aligned_gat = gat_embeddings @ R
    
    Returns:
        R: Rotation matrix [embedding_dim, embedding_dim] or None if not found
        
    Also sets instance variables:
        self.procrustes_use_centering: bool
        self.procrustes_hpl_mean: np.ndarray or None
        self.procrustes_gat_mean: np.ndarray or None
    """
    try:
        with h5py.File(self.model_path, 'r') as f:
            if 'procrustes/rotation_matrix' not in f:
                self._log("  ⚠️  No rotation matrix found in checkpoint")
                return None
            
            R = f['procrustes/rotation_matrix'][:]
            
            # Validate rotation matrix properties
            embedding_dim = R.shape[0]
            if R.shape[0] != R.shape[1]:
                self._log(f"  ⚠️  Invalid rotation matrix shape: {R.shape}")
                return None
            
            # Check orthogonality (R.T @ R should be identity)
            RtR = R.T @ R
            identity = np.eye(embedding_dim)
            orthogonality_error = np.linalg.norm(RtR - identity, 'fro')
            
            if orthogonality_error > 1e-3:
                self._log(f"  ⚠️  Rotation matrix may not be orthogonal (error: {orthogonality_error:.2e})")
            
            # Load alignment metrics and centering data if available
            self.procrustes_use_centering = False
            self.procrustes_hpl_mean = None
            self.procrustes_gat_mean = None
            
            if 'procrustes' in f:
                procrustes_grp = f['procrustes']
                
                # Load alignment metrics
                if 'error_after' in procrustes_grp.attrs:
                    error_after = procrustes_grp.attrs['error_after']

                
                # Check for mean centering (enhanced Procrustes)
                # Support both explicit attribute and auto-detection from datasets
                use_centering_attr = procrustes_grp.attrs.get('use_centering', False)
                has_mean_datasets = 'hpl_mean' in procrustes_grp and 'gat_mean' in procrustes_grp
                
                # Enable mean centering if EITHER the attribute is True OR datasets exist
                # This provides backward compatibility for checkpoints that saved means
                # but didn't explicitly set the use_centering attribute
                if use_centering_attr or has_mean_datasets:
                    if has_mean_datasets:
                        self.procrustes_hpl_mean = procrustes_grp['hpl_mean'][:].astype(np.float32)
                        self.procrustes_gat_mean = procrustes_grp['gat_mean'][:].astype(np.float32)
                        self.procrustes_use_centering = True
                    else:
                         pass
                else:
                     pass
            
            return R.astype(np.float32)
            
    except Exception as e:
        self._log(f"  ⚠️  Failed to load rotation matrix: {e}")
        import traceback
        traceback.print_exc()
        return None
def _read_checkpoint_metadata(model_path) -> Dict[str, Any]:
    """Read and validate the portable inference contract without loading weights.

    The checkpoint owns gene columns, species order, and normalization. The
    packaged gene resources are used by standalone gene-alignment and QC helpers.
    """
    import json

    def decode(value):
        return value.decode('utf-8') if isinstance(value, bytes) else value

    with h5py.File(model_path, 'r') as f:
        attrs = f['config'].attrs
        config = json.loads(decode(attrs['model_config']))
        if config.get('inference_format_version') != 1:
            raise ValueError(
                'This package requires a portable six-species checkpoint '
                '(inference_format_version=1). Pass the new model path to '
                'HECTOR(model_path=...). The retired human/mouse checkpoint '
                'cannot supply the required native gene mappings.'
            )
        encoder = config['encoder_config']
        if not encoder.get('use_gene_tokens'):
            raise ValueError('Only gene-token checkpoints are supported.')
        width = int(encoder['num_genes'])
        backbone = int(encoder['num_backbone_genes'])
        if not 0 < backbone <= width:
            raise ValueError('Invalid checkpoint backbone/input width.')
        for key in ('num_genes', 'input_dim'):
            if config.get(key, width) != width:
                raise ValueError(f'Checkpoint {key} disagrees with encoder_config.')

        decoder = config['decoder']
        species = list(decoder['species_names'])
        if not species or len(set(species)) != len(species):
            raise ValueError('Checkpoint species order must be nonempty and unique.')
        if any(_canonical_species(sp) != sp for sp in species):
            raise ValueError('Checkpoint species names must be canonical names.')
        sizes = decoder['gene_sizes']
        if len(sizes) != len(species):
            raise ValueError('Native gene sizes do not match checkpoint species.')
        native_ids, native_columns = {}, {}
        for sp, size in zip(species, sizes):
            genes = f[f'config/native_gene_ids/{sp}'].asstr()[:].tolist()
            columns = f[f'config/native_gene_columns/{sp}'][:]
            if (columns.ndim != 1 or not np.issubdtype(columns.dtype, np.integer)
                    or len(genes) != size or len(columns) != size or not size
                    or len(set(genes)) != size or any(not g for g in genes)
                    or np.any(columns < 0) or np.any(columns >= width)
                    or np.any(np.diff(columns) <= 0)):
                raise ValueError(f'Invalid native gene/column mapping for {sp}.')
            native_ids[sp] = genes
            native_columns[sp] = columns.astype(np.int32, copy=False)

        assets = f['weights/vgae_encoder/gene_token_assets']
        mapping = assets['col_to_row'][:]
        table = assets['protein_embeddings']
        if (mapping.shape != (len(species), width)
                or not np.issubdtype(mapping.dtype, np.integer)
                or table.ndim != 2 or not np.issubdtype(table.dtype, np.floating)
                or mapping.min() < 0 or mapping.max() >= table.shape[0]):
            raise ValueError('Gene-token assets disagree with checkpoint dimensions.')
        # Protein row numbers are opaque indices, not gene-presence flags.
        # Row 0 can represent a real protein; absent genes can use a final zero row.
        # Native gene columns above define availability;
        # the lookup only needs valid indices into this checkpoint's table.

        normalization = json.loads(decode(attrs['normalization_config']))
        if normalization.get('n_backbone') != backbone:
            raise ValueError('Normalization and encoder backbone widths disagree.')
        if normalization.get('method') not in ('log1p_norm', 'none'):
            raise ValueError('Unsupported checkpoint normalization method.')
        if (not np.isfinite(normalization.get('target_median', np.nan))
                or normalization['target_median'] <= 0
                or not 0 <= normalization.get('clip_percentile', -1) <= 100):
            raise ValueError('Invalid checkpoint normalization parameters.')
        if 'memory_bank' in f:
            bank = f['memory_bank']
            size = int(bank['mem_weight_1'][()])
            if (int(bank.attrs['feature_dim']) != backbone
                    or bank['mem_weight_2'].shape[1] != backbone
                    or bank['mem_weight_3'].shape[1] != encoder['shared_dim']
                    or not 0 <= size <= bank['mem_weight_2'].shape[0]
                    or bank['mem_weight_3'].shape[0] != bank['mem_weight_2'].shape[0]):
                raise ValueError('Memory bank disagrees with encoder dimensions.')

        return {
            'checkpoint_version': decode(f.attrs.get('checkpoint_version')),
            'model_config': config,
            'normalization_config': normalization,
            'species_names': species,
            'native_gene_ids': native_ids,
            'native_gene_columns': native_columns,
            'layout_hash': decoder['layout_hash'],
            'procrustes_aligned': bool(attrs.get('procrustes_aligned', False)),
            'alignment_quality': decode(attrs.get('alignment_quality')),
            'alignment_cosine_sim': attrs.get('alignment_cosine_sim'),
        }


def _load_checkpoint_metadata(self) -> Dict[str, Any]:
    return _read_checkpoint_metadata(self.model_path)


# ---------------------------------------------------------------------------
# HECTOR batch-size and memory estimators
# ---------------------------------------------------------------------------
def _estimate_encoder_memory_terms(
    n_genes: int,
    shared_dim: int,
    k_neighbors: int,
    graph_attention_type: str,
    latent_dim: int,
    n_anchors: int,
    *,
    is_gene_token: bool = False,
    gene_token_k: int = 2048,
    token_dim: int = 1024,
) -> "Tuple[int, int]":
    """Estimate (fixed_bytes, per_cell_bytes) for one encoder forward pass.

    Peak VRAM model:
        peak ≈ fixed_bytes + B × per_cell_bytes

    ``fixed_bytes`` covers anchor state and transients independent of B.
    ``per_cell_bytes`` covers input, hidden, attention, and output tensors
    that scale linearly with B. Gene-token models use saved anchor hidden
    vectors instead of re-encoding anchors.

    Returns:
        (fixed_bytes, per_cell_bytes) tuple.
    """
    # Gene-token models use saved anchor hidden vectors. Account for their
    # resident state, token features, cosine search, and input copies.
    # This simplified CUDA estimate has not been hardware-validated.
    if is_gene_token:
        fixed_bytes = int(n_anchors * shared_dim * 4)
        per_cell_bytes = int(gene_token_k * token_dim * 4 + n_anchors * 4
                             + 2 * n_genes * 4)
        return fixed_bytes, per_cell_bytes

    d = shared_dim
    k = k_neighbors

    # ---------------- fixed_bytes: anchor-side, constant in B ----------------
    # Include raw and hidden anchor state plus a conservative factor
    # for the one-time anchor-encoding transient.
    _ANCHOR_TRANSIENT_FACTOR = 2.3
    fixed_bytes = int(n_anchors * (n_genes + d) * 4 * _ANCHOR_TRANSIENT_FACTOR)

    # ---------------- per_cell_bytes: linear in B ----------------
    input_bytes = n_genes * 4
    hidden_bytes = 2 * d * 4

    if graph_attention_type == 'transformer':
        attn_bytes = k * d * 4 * 5
    else:
        attn_bytes = k * d * 4 * 4

    output_bytes = (latent_dim + 768) * 4 * 2

    per_cell_bytes = input_bytes + hidden_bytes + attn_bytes + output_bytes

    return fixed_bytes, per_cell_bytes


def _estimate_encoder_memory_terms_metal(
    n_genes: int,
    n_anchors: int,
    shared_dim: int,
    *,
    is_gene_token: bool = False,
    gene_token_k: int = 2048,
    token_dim: int = 1024,
) -> "Tuple[int, int]":
    """Estimate (fixed_floor_bytes, per_cell_bytes) for the MLX/Metal encoder.

    The gene-token branch counts saved anchor hidden vectors as the fixed
    cost, then estimates a FiLM token block, a full anchor-similarity row,
    and two dense input copies per query cell. The non-token branch estimates
    an anchor working-set floor and the similarity/input costs per cell.

    This simplified model omits a separate attention workspace term. Staged
    gene-token planning uses ``_estimate_gene_token_stage_memory`` instead.

    Returns:
        (fixed_floor_bytes, per_cell_bytes) tuple.
    """
    if is_gene_token:
        fixed_floor = int(n_anchors * shared_dim * 4)
        per_cell = int(gene_token_k * token_dim * 4 + n_anchors * 4
                       + 2 * n_genes * 4)
        return fixed_floor, per_cell
    resident = n_anchors * (n_genes + shared_dim) * 4
    fixed_floor = int(resident * _METAL_FLOOR_TRANSIENT_FACTOR)
    per_cell = (n_anchors + 2 * n_genes) * 4
    return fixed_floor, per_cell


@dataclass(frozen=True)
class GeneTokenMemoryInputs:
    """Tensor dimensions needed by the staged gene-token memory model."""

    backend: str
    n_cells: int
    n_genes: int
    n_backbone: int
    n_anchors: int
    shared_dim: int
    token_dim: int
    gene_token_k: int
    k_neighbors: int
    num_attention_heads: int
    latent_dim: int
    persistent_weight_bytes: int = 0
    film_table_rows: int = 0
    dtype_bytes: int = 4


@dataclass(frozen=True)
class GeneTokenMemoryEstimate:
    """Peak bytes for each independently executed encoder stage."""

    knn_bytes: int
    token_bytes: int
    graph_bytes: int
    device_peak_bytes: int
    system_increment_bytes: int
    limiting_stage: str
    largest_single_buffer_bytes: int = 0


@dataclass(frozen=True)
class GeneTokenExecutionPlan:
    """Bounded execution parameters selected for one gene-token encode."""

    backend: str
    batch_size: int
    token_chunk_size: int
    knn_candidate_tile: int
    knn_mode: str
    projected_peak_bytes: int
    projected_system_increment_bytes: int
    device_budget_bytes: int
    system_budget_bytes: "int | None"
    limiting_stage: str
    feasible: bool


@dataclass(frozen=True)
class GritMemoryPlan:
    """Bounded query/candidate tiles for exact MLX GRIT k-NN."""

    query_tile_size: int
    candidate_tile_size: int
    projected_peak_bytes: int
    projected_system_increment_bytes: int
    device_budget_bytes: int
    system_budget_bytes: "int | None"
    feasible: bool


def _estimate_grit_tiled_memory(
    *,
    n_cells: int,
    n_features: int,
    n_queries: int,
    k_neighbors: int,
    query_tile_size: int,
    candidate_tile_size: int,
    active_device_bytes: int = 0,
) -> Tuple[int, int]:
    """Return total Metal peak and additional unified memory for tiled GRIT."""
    n = max(1, int(n_cells))
    features = max(1, int(n_features))
    queries = max(1, min(int(n_queries), n))
    q = max(1, min(int(query_tile_size), queries))
    c = max(1, min(int(candidate_tile_size), n))
    k = max(1, min(int(k_neighbors) + 1, n))
    f = 4

    # Host tiles are materialized from CSR/dense input. MLX owns separate tile
    # arrays plus similarity/partition transients; running top-k state stays on
    # the host between candidate tiles.
    host_tiles = (q + c) * features * f
    device_tiles = (q + c) * features * f
    similarity_work = 4 * q * c * f
    topk_device = 12 * q * k * f
    allocator_guard = 8 * 1024 ** 2
    device_increment = device_tiles + similarity_work + topk_device + allocator_guard

    row_norms = n * f
    running_topk = queries * k * (f + 4)
    edge_outputs = 4 * queries * max(1, int(k_neighbors)) * 4
    system_increment = (
        host_tiles + device_increment + row_norms + running_topk + edge_outputs
    )
    return (
        int(max(0, int(active_device_bytes)) + device_increment),
        int(system_increment),
    )


def _plan_grit_execution(
    *,
    n_cells: int,
    n_features: int,
    n_queries: int,
    k_neighbors: int,
    active_device_bytes: int,
    device_budget_bytes: int,
    system_budget_bytes: "int | None" = None,
    minimum_tile_size: int = 64,
    max_single_alloc_bytes: "int | None" = None,
) -> GritMemoryPlan:
    """Choose the largest feasible tile area from a discrete candidate grid.

    Device, system-memory, and single-buffer limits are checked against
    analytical estimates. If no candidate fits, return the smallest tested
    fallback with feasible=False.
    """
    n = max(1, int(n_cells))
    queries = max(1, min(int(n_queries), n))
    minimum = max(1, int(minimum_tile_size))
    caps = (8192, 4096, 2048, 1024, 512, 256, 128, minimum)
    query_tiles = sorted({min(queries, max(1, value)) for value in caps}, reverse=True)
    candidate_tiles = sorted({min(n, max(1, value)) for value in caps}, reverse=True)
    candidates = sorted(
        ((q * c, c, q) for q in query_tiles for c in candidate_tiles),
        reverse=True,
    )

    fallback = None
    for _area, c, q in candidates:
        peak, system_increment = _estimate_grit_tiled_memory(
            n_cells=n,
            n_features=n_features,
            n_queries=queries,
            k_neighbors=k_neighbors,
            query_tile_size=q,
            candidate_tile_size=c,
            active_device_bytes=active_device_bytes,
        )
        # Largest single mx.array in build_knn_graph_cells_mlx: the q x c
        # similarity, or a densified query/candidate tile (q or c) x n_features.
        largest_single = max(q * c, c * n_features, q * n_features) * 4
        feasible = (
            peak <= int(device_budget_bytes)
            and (
                system_budget_bytes is None
                or system_increment <= int(system_budget_bytes)
            )
            and (
                max_single_alloc_bytes is None
                or largest_single <= int(max_single_alloc_bytes)
            )
        )
        plan = GritMemoryPlan(
            query_tile_size=q,
            candidate_tile_size=c,
            projected_peak_bytes=peak,
            projected_system_increment_bytes=system_increment,
            device_budget_bytes=int(device_budget_bytes),
            system_budget_bytes=(
                None if system_budget_bytes is None else int(system_budget_bytes)
            ),
            feasible=bool(feasible),
        )
        fallback = plan
        if feasible:
            return plan

    return fallback


def _estimate_gene_token_stage_memory(
    inputs: GeneTokenMemoryInputs,
    *,
    batch_size: int,
    token_chunk_size: int,
    knn_candidate_tile: int,
    knn_mode: str = "tiled",
) -> GeneTokenMemoryEstimate:
    """Estimate the maximum of k-NN, token, and graph-attention stages.

    MLX estimates follow tensor shapes. CUDA adds stage-specific workspace
    factors and fixed overhead. The result also estimates selected single
    buffer sizes and additional host or unified memory.
    """
    f = max(1, int(inputs.dtype_bytes))
    B = max(1, min(int(batch_size), int(inputs.n_cells)))
    C = max(1, min(int(token_chunk_size), B))
    A = max(0, int(inputs.n_anchors))
    G = max(0, int(inputs.n_genes))
    G_backbone = max(0, int(inputs.n_backbone))
    D = max(1, int(inputs.shared_dim))
    T = max(1, int(inputs.token_dim))
    K = max(1, min(int(inputs.gene_token_k), G))
    graph_k = max(1, int(inputs.k_neighbors) + 1)  # explicit self-loop
    heads = max(1, int(inputs.num_attention_heads))
    latent = max(1, int(inputs.latent_dim))
    tile = max(1, min(int(knn_candidate_tile), max(A, 1)))

    raw_anchors = A * G_backbone * f
    hidden_anchors = A * D * f
    weights = max(0, int(inputs.persistent_weight_bytes))
    base = weights + raw_anchors + hidden_anchors

    # Query normalization and MLX argpartition keep several tile-width arrays
    # live. Counting three similarity-width buffers is conservative for the
    # on-device merge and also covers candidate-index storage.
    query_work = 2 * B * G_backbone * f
    candidate_copies = 1 if tile >= max(A, 1) else 2
    candidate_transpose = candidate_copies * tile * G_backbone * f
    similarity_work = 3 * B * tile * f
    topk_work = 10 * B * graph_k * f

    # Five token-width arrays cover FiLM gamma, beta, token/normalization, and
    # within-cell K/V projections. Smaller sorting/index arrays are explicit.
    token_per_cell = (
        5 * K * T * f
        + 2 * G * f
        + 4 * K * f
        + 3 * D * f
    )
    token_work = C * token_per_cell + B * (G + D) * f

    # h_full + normalized h_full, neighbor gather and K/V-like projections,
    # attention logits, and the two variational output heads.
    graph_nodes = (A + B) * D * f
    graph_work = (
        2 * graph_nodes
        + 3 * B * graph_k * D * f
        + 4 * B * D * f
        + 2 * B * graph_k * heads * f
        + 2 * B * latent * f
    )

    # Selected buffer sizes for the planner's single-allocation estimate.
    # Compare this estimate with mlx_max_alloc() as well as the total budget.
    largest_single_buffer = max(
        raw_anchors,              # A x G_backbone anchor bank (resident)
        tile * G_backbone * f,    # one candidate tile
        B * tile * f,             # B x tile similarity block
        B * G_backbone * f,       # normalized query batch
        B * G * f,                # dense input batch
        B * graph_k * D * f,      # graph-attention neighbour gather
        (A + B) * D * f,          # concatenated [batch, anchors] node features
    )

    if inputs.backend == "cuda":
        # Include allocator and kernel-workspace allowances for each CUDA stage.
        workspace_fixed = 2816 * 1024 ** 2   # CUDA context, library workspace, and allocator floor
        workspace_factor = 1.6               # kNN / graph transient multiplier
        token_factor = 1.8                   # gene-token transient multiplier
        h_anchor_new = hidden_anchors
        full_anchor_cache = raw_anchors if knn_mode == "full" else 0
        # Keep one normalized candidate tile live around the matrix multiplication.
        tiled_candidates = 0 if knn_mode == "full" else tile * G_backbone * f
        film_projection = max(0, int(inputs.film_table_rows)) * 2 * T * f
        # Full mode is single-shot: one [B, n_cand] similarity matrix. Tiled mode
        # keeps ~2 similarity-width blocks (query*candidate + running merge).
        cuda_similarity = (1 if knn_mode == "full" else 2) * B * tile * f
        knn_transient = query_work + cuda_similarity + topk_work
        token_transient = film_projection + token_work
        graph_transient = graph_work
        knn_bytes = (
            h_anchor_new + full_anchor_cache + tiled_candidates
            + int(knn_transient * workspace_factor) + workspace_fixed
        )
        token_bytes = h_anchor_new + full_anchor_cache + int(token_transient * token_factor) + workspace_fixed
        graph_bytes = h_anchor_new + full_anchor_cache + int(graph_transient * workspace_factor) + workspace_fixed
        allocator_guard = 8 * 1024 ** 2
        knn_bytes += allocator_guard
        token_bytes += allocator_guard
        graph_bytes += allocator_guard
        system_increment = inputs.n_cells * (latent * f + f)
    else:
        # Uploading the NumPy anchor bank to MLX transiently holds two raw-anchor
        # allocations even when subsequent cosine search is candidate-tiled.
        anchor_upload_floor = (
            weights + 2 * raw_anchors + hidden_anchors + A * f + 1024 ** 2
        )
        tiled_knn = base + candidate_transpose + query_work + similarity_work + topk_work
        knn_bytes = max(anchor_upload_floor, tiled_knn)
        token_bytes = base + token_work
        graph_bytes = base + graph_work
        allocator_guard = 8 * 1024 ** 2
        knn_bytes += allocator_guard
        token_bytes += allocator_guard
        graph_bytes += allocator_guard
        # Unified-memory growth includes anchor pages retained between stages
        # plus per-batch graph-gather and dense buffers:
        #
        #   phys(B,C,tile) = anchor_floor                         (batch-independent)
        #                  + B * (graph gather + dense buffer)    (ADDITIVE per cell)
        #                  + max(token growth, kNN-tile growth)   (sequential -> MAX)
        #                  + host outputs + fixed overhead
        #
        # Token and k-NN stages run sequentially, so take the maximum extra
        # growth rather than summing both. Charge only growth above the
        # minimum token chunk and candidate tile.
        graph_gather_per_cell = (
            3 * graph_k * D + 2 * graph_k * heads + 6 * D + 2 * latent
        ) * f
        dense_buffer_per_cell = G * f
        per_batch_additive = B * (graph_gather_per_cell + dense_buffer_per_cell)
        token_growth = max(0, C - _MLX_TOKEN_CHUNK_FLOOR) * 4 * K * T * f
        tile_growth = max(0, tile - _MLX_TILE_FLOOR) * G_backbone * f * 2
        speed_knob_growth = max(token_growth, tile_growth)
        host_outputs = inputs.n_cells * (latent * f + f)
        system_increment = (
            anchor_upload_floor
            + per_batch_additive
            + speed_knob_growth
            + host_outputs
            + _METAL_UNIFIED_OVERHEAD_BYTES
        )

    stage_bytes = {
        "knn": int(knn_bytes),
        "tokens": int(token_bytes),
        "graph": int(graph_bytes),
    }
    limiting_stage = max(stage_bytes, key=stage_bytes.get)
    peak = stage_bytes[limiting_stage]
    return GeneTokenMemoryEstimate(
        knn_bytes=stage_bytes["knn"],
        token_bytes=stage_bytes["tokens"],
        graph_bytes=stage_bytes["graph"],
        device_peak_bytes=int(peak),
        system_increment_bytes=int(system_increment),
        limiting_stage=limiting_stage,
        largest_single_buffer_bytes=int(largest_single_buffer),
    )


def _plan_gene_token_execution(
    inputs: GeneTokenMemoryInputs,
    *,
    device_budget_bytes: int,
    system_budget_bytes: "int | None" = None,
    batch_cap: "int | None" = None,
    explicit_batch_size: "int | None" = None,
    max_single_alloc_bytes: "int | None" = None,
) -> GeneTokenExecutionPlan:
    """Select an outer batch, token chunk, and k-NN tile using memory estimates.

    Automatic sizing searches batches up to n_cells and an optional batch_cap
    using the smallest chunk and tile, then grows these from discrete grids.
    explicit_batch_size overrides the automatic batch search. If even the
    minimum plan exceeds a limit, return it with feasible=False.
    Chunking and tiling preserve the algorithm, while floating-point rounding
    can depend on execution shape.
    """
    n_cells = max(1, int(inputs.n_cells))
    upper = n_cells if batch_cap is None else min(n_cells, max(1, int(batch_cap)))

    chunk_caps = (512, 384, 256, 192, 160, 128, 96, 64, 32)
    tile_caps = (inputs.n_anchors, 65536, 32768, 16384, 8192, 4096)
    tile_min = max(1, min(t for t in tile_caps if t > 0))
    modes = ("full", "tiled") if inputs.backend == "cuda" else ("tiled",)

    def estimate(B, C, tile, mode):
        return _estimate_gene_token_stage_memory(
            inputs, batch_size=B, token_chunk_size=C,
            knn_candidate_tile=tile, knn_mode=mode)

    def fits(est):
        return (
            est.device_peak_bytes <= int(device_budget_bytes)
            and (
                system_budget_bytes is None
                or est.system_increment_bytes <= int(system_budget_bytes)
            )
            and (
                max_single_alloc_bytes is None
                or est.largest_single_buffer_bytes <= int(max_single_alloc_bytes)
            )
        )

    def floor_fits(B):
        # Minimum-peak config for this B: smallest chunk, smallest tile, tiled.
        return fits(estimate(B, min(32, B), tile_min, "tiled"))

    # 1) Choose the outer batch.
    if explicit_batch_size is not None:
        B = min(n_cells, max(1, int(explicit_batch_size)))
    elif not floor_fits(1):
        B = 1  # even the minimum does not fit; return the nonblocking minimum
    else:
        lo, hi = 1, upper
        while lo < hi:                        # largest B with floor_fits(B)
            mid = (lo + hi + 1) // 2
            if floor_fits(mid):
                lo = mid
            else:
                hi = mid - 1
        B = lo
        if B >= 32 and B < n_cells:           # 32-align only when memory-limited
            B -= B % 32

    # 2) Grow chunk and tile into the headroom under the binding stage. Token
    # depends only on C, k-NN only on tile, so they are maximized independently.
    best_C = min(32, B)
    for C in sorted({min(B, c) for c in chunk_caps}, reverse=True):
        if fits(estimate(B, C, tile_min, "tiled")):
            best_C = C
            break
    best_tile, best_mode = tile_min, "tiled"
    for mode in modes:
        candidate_tiles = ([inputs.n_anchors] if mode == "full"
                           else sorted({min(max(inputs.n_anchors, 1), t)
                                        for t in tile_caps}, reverse=True))
        for tile in candidate_tiles:
            if fits(estimate(B, best_C, tile, mode)):
                best_tile, best_mode = tile, mode
                break
        else:
            continue
        break

    est = estimate(B, best_C, best_tile, best_mode)
    return GeneTokenExecutionPlan(
        backend=inputs.backend, batch_size=B, token_chunk_size=best_C,
        knn_candidate_tile=best_tile, knn_mode=best_mode,
        projected_peak_bytes=est.device_peak_bytes,
        projected_system_increment_bytes=est.system_increment_bytes,
        device_budget_bytes=int(device_budget_bytes),
        system_budget_bytes=(None if system_budget_bytes is None
                             else int(system_budget_bytes)),
        limiting_stage=est.limiting_stage, feasible=bool(fits(est)),
    )


def _estimate_classification_bytes_per_row(
    self,
    use_full_ontology: Optional[bool] = None,
) -> int:
    """Estimate peak GPU memory per cell for one classification batch."""

    if use_full_ontology is None:
        use_full_ontology = self.config.use_full_ontology

    num_full_classes = len(self.celltype_gat.full_classes)
    if use_full_ontology:
        num_active_classes = num_full_classes
    else:
        num_active_classes = len(self.celltype_gat.classes)

    num_seen_classes = num_active_classes
    if self.aux_classifier is not None:
        num_seen_classes = int(
            self.aux_classifier.ancestry_matrix.shape[0]
        )

    embedding_dim = 768

    # Dense float score tensors dominate peak usage:
    # generalist, expert, profile, base, filtered, and one propagated mask-like
    score_bytes = num_full_classes * 4 * 6
    seen_bytes = num_seen_classes * 4 * 2
    mask_bytes = num_active_classes * 4
    embedding_bytes = embedding_dim * 4 * 6

    return score_bytes + seen_bytes + mask_bytes + embedding_bytes
def _resolve_stream_batch_size(
    self,
    n_rows: int,
    fixed_bytes: int,
    per_cell_bytes: int,
    *,
    overhead_factor: float = 2.0,
    default_batch: int = 50000,
    silent: bool = False,
) -> int:
    """Use an explicit configured batch, GPU sizing, or the default, in that order.

    GPU memory model: ``peak ≈ fixed_bytes + B × per_cell_bytes × overhead_factor``.
    Pass ``fixed_bytes=0`` for ops with no anchor-style fixed cost.
    """

    if self.config.batch_size is not None:
        return min(int(self.config.batch_size), n_rows)

    has_gpu = bool(tf.config.list_physical_devices("GPU")) if tf is not None else False
    if has_gpu:
        try:
            return self._get_gpu_aware_batch_size(
                n_rows,
                fixed_bytes,
                per_cell_bytes,
                overhead_factor=overhead_factor,
                default_batch=default_batch,
                silent=silent,
            )
        except Exception:
            pass

    return min(default_batch, n_rows)

# Fallback Metal floor for the one-time anchor-encoding working set,
# used when a live peak-memory probe is unavailable.
_METAL_FLOOR_TRANSIENT_FACTOR = float(
    os.environ.get("HECTOR_METAL_FLOOR_FACTOR", "2.6")
)

# Fixed unified-memory allowance for Metal context and pages retained
# between stages.
_METAL_UNIFIED_OVERHEAD_BYTES = int(
    os.environ.get("HECTOR_METAL_UNIFIED_OVERHEAD_BYTES", str(614 * 1024 ** 2))
)

# The planner's minimum token chunk and candidate tile. Both sit under the anchor
# floor, so only growth PAST them adds to the phys model (see the MLX branch of
# _estimate_gene_token_stage_memory). Kept in sync with _plan_gene_token_execution.
_MLX_TOKEN_CHUNK_FLOOR = 32
_MLX_TILE_FLOOR = 4096


def _get_macos_available_bytes() -> "int | None":
    """Reclaimable bytes on macOS via Mach host_statistics64 (ctypes, no subprocess)."""
    try:
        import ctypes
        import ctypes.util

        class _vm_statistics64(ctypes.Structure):
            _fields_ = [
                ("free_count", ctypes.c_uint32),
                ("active_count", ctypes.c_uint32),
                ("inactive_count", ctypes.c_uint32),
                ("wire_count", ctypes.c_uint32),
                ("zero_fill_count", ctypes.c_uint64),
                ("reactivations", ctypes.c_uint64),
                ("pageins", ctypes.c_uint64),
                ("pageouts", ctypes.c_uint64),
                ("faults", ctypes.c_uint64),
                ("cow_faults", ctypes.c_uint64),
                ("lookups", ctypes.c_uint64),
                ("hits", ctypes.c_uint64),
                ("purges", ctypes.c_uint64),
                ("purgeable_count", ctypes.c_uint32),
                ("speculative_count", ctypes.c_uint32),
                ("decompressions", ctypes.c_uint64),
                ("compressions", ctypes.c_uint64),
                ("swapins", ctypes.c_uint64),
                ("swapouts", ctypes.c_uint64),
                ("compressor_page_count", ctypes.c_uint32),
                ("throttled_count", ctypes.c_uint32),
                ("external_page_count", ctypes.c_uint32),
                ("internal_page_count", ctypes.c_uint32),
                ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
            ]

        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.mach_host_self.restype = ctypes.c_uint32
        host = libc.mach_host_self()

        stats = _vm_statistics64()
        count = ctypes.c_uint32(ctypes.sizeof(stats) // 4)
        HOST_VM_INFO64 = 4
        ret = libc.host_statistics64(
            host, HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count),
        )
        if ret != 0:
            return None

        page_size = os.sysconf("SC_PAGE_SIZE")
        # Free, inactive, and purgeable pages are reclaimable memory.
        # Include inactive file-cache pages so cache growth does not shrink the budget.
        reclaimable = (
            stats.free_count + stats.inactive_count + stats.purgeable_count
        )
        return int(reclaimable) * page_size
    except Exception:
        return None

@functools.lru_cache(maxsize=1)
def mlx_device_limits() -> "Tuple[int, int, int]":
    """Return cached (working_set, max_buffer, total) bytes for the Apple GPU.

    Reads the GPU limits from ``mx.device_info()`` and caches them for the process.

    * working_set -- ``max_recommended_working_set_size`` (Metal working-set cap).
    * max_buffer  -- ``max_buffer_length`` (hard cap on any SINGLE ``mx.array``).
    * total       -- ``memory_size`` (total unified RAM).

    Off-Metal fallback (MLX unavailable): ``total_phys/3.1`` for both caps and the
    real total, so non-Apple hosts still get a finite, conservative budget. Cached
    for the process.
    """
    try:
        import mlx.core as mx
        info = mx.device_info()
        working_set = int(info["max_recommended_working_set_size"])
        max_buffer = int(info.get("max_buffer_length") or working_set)
        total = int(info.get("memory_size") or 0)
        if total <= 0:
            total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        return working_set, max_buffer, total
    except Exception:
        total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        conservative = int(total // 3.1)
        return conservative, conservative, total


def mlx_max_alloc(safety: float = 0.90) -> int:
    """Byte cap on any single ``mx.array`` -- ``safety * max_buffer_length``.

    Metal refuses a single buffer larger than ``max_buffer_length`` even when the
    total budget looks fine. Tile/batch planners compare estimated single-array
    sizes against this cap when selecting an execution plan.
    """
    return int(mlx_device_limits()[1] * float(safety))


def _mlx_budget_bytes(
    *,
    working_set: int,
    available: int,
    active: int,
    cache: int,
    safety_fraction: float,
    reserved_system_bytes: int = 0,
    cache_add_back: bool = False,
) -> "Tuple[int, int, int]":
    """Pure core of the MLX budget: ``(device_cap, system_budget, active)`` bytes.

    The caller supplies measurements; this function performs no device or OS I/O.

    * ``device_cap``    = ``working_set * safety``
    * ``system_budget`` = ``max(0, available (+cache if cache_add_back) - reserved) * safety``

    ``cache_add_back`` defaults to False. When True, MLX's reusable free pool
    is counted as available, permitting larger tiles or batches.
    """
    device_cap = int(int(working_set) * float(safety_fraction))
    usable = int(available) + (int(cache) if cache_add_back else 0)
    system_budget = int(
        max(0, usable - int(reserved_system_bytes)) * float(safety_fraction)
    )
    return device_cap, system_budget, int(active)


def mlx_unified_budget_bytes(
    safety_fraction: float = 0.85,
    *,
    reserved_bytes: int = 0,
    op_host_bytes: int = 0,
    cache_add_back: bool = False,
) -> int:
    """Scalar free-aware unified-memory budget for the next MLX op, in bytes.

    ``max(0, min(device_cap, system_budget) - op_host_bytes)`` -- the same device
    and free-memory sources as ``_mlx_memory_budgets``, collapsed to ONE number for
    sizers that want a single ceiling (the t-SNE refusal gate, the exact-kNN and
    repulsive distance tiles) rather than the (device, system, active) tuple the
    staged planners consume. Nets out resident memory via the live macOS-free
    reading, so a loaded host shrinks the budget automatically.
    """
    working_set, _max_buffer, _total = mlx_device_limits()
    available = _get_macos_available_bytes()
    if available is None:
        available = working_set
    device_cap, system_budget, _ = _mlx_budget_bytes(
        working_set=working_set,
        available=int(available),
        active=0,
        cache=0,
        safety_fraction=safety_fraction,
        reserved_system_bytes=reserved_bytes,
        cache_add_back=cache_add_back,
    )
    return max(0, min(device_cap, system_budget) - int(op_host_bytes))


def _vram_bytes_held_by_other_processes() -> "int | None":
    """Bytes on GPU 0 held by processes other than this one, or None if unknowable.

    Needed because the card's total allocation says nothing about who allocated it.
    Without this split, memory belonging to another process is indistinguishable from
    our own TensorFlow arena, and gets offered to the batch sizer as reusable space.

    Returns None rather than 0 when the driver cannot be asked or declines to answer
    for any process; callers must treat that as "cannot attribute" and fall back to
    the free reading alone, never as "no other process is running".

    The driver declines in two situations worth naming. On Windows the display driver
    model in normal desktop use does not expose per-process memory at all, so every
    process comes back unreadable. And inside a container the process identifiers here
    are the host's, which will not match ours, so this process counts itself among the
    others -- an understatement of available memory, which is the safe direction, but
    the reason the numbers can look odd there.
    """
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        card_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    except Exception:
        return None
    import os
    me = os.getpid()
    total = 0
    for proc in procs:
        if proc.pid == me:
            continue
        used = proc.usedGpuMemory
        # Unreadable usage arrives as an unsigned -1, not as None: the field is a
        # plain c_ulonglong with no sentinel handling. Anything at or above the card's
        # own capacity is that sentinel, not a reading. One unreadable process makes
        # the whole sum a guess, so give up rather than under-report.
        if used is None or int(used) >= card_bytes:
            return None
        total += int(used)
    return total


def _available_vram_bytes(*, allocator: str = "tf") -> "int | None":
    """Free VRAM in bytes for the NEXT allocation, or None if no CUDA/cupy.

    The "free VRAM" that matters depends on which allocator is about to allocate,
    so this is deliberately not one number:

    * ``allocator="tf"``  -> ``OS-free + TF-arena-slack``. TF grows and caches
      its own arena (memory-growth mode), so what it can serve includes cached
      slack that OS-free alone misses. The two pools are disjoint -- memory nobody
      has claimed, and memory we hold but are not using -- so they add.
    * ``allocator="rmm"`` -> release cupy's cached pool (``free_all_blocks``) first,
      then OS-free. This path is intended for cuML / cuGraph allocations; TF's private
      arena is NOT reusable by them, so arena slack must NOT be added here.

    Returns None when cupy/CUDA is unavailable -- callers keep their own fallback.
    """
    try:
        import cupy as cp
    except Exception:
        return None
    try:
        if allocator == "rmm":
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            free, _total = cp.cuda.Device().mem_info
            return int(free)
        # allocator == "tf": add TF's reusable arena slack.
        tf_current = 0
        try:
            import tensorflow as _tf
            tf_current = _tf.config.experimental.get_memory_info("GPU:0").get("current", 0)
        except Exception:
            pass
        free, total = cp.cuda.Device().mem_info
        # Subtract other processes' allocations before counting reusable arena
        # slack. When ownership is unknown, use only the free reading.
        held_by_others = _vram_bytes_held_by_other_processes()
        if held_by_others is None:
            return int(free)
        held_by_us = max(0, (total - free) - held_by_others)
        arena_slack = max(0, held_by_us - tf_current)
        return int(free) + arena_slack
    except Exception:
        return None


def _get_gpu_aware_batch_size(self, n_rows: int, fixed_bytes: int,
                               per_cell_bytes: int,
                               overhead_factor: float = 2.0,
                               default_batch: int = 50000,
                               silent: bool = False) -> int:
    """Size the next TensorFlow batch from available VRAM.

    Available memory combines OS-free VRAM and reusable slack in this process's
    TensorFlow arena. Subtract other processes' allocations before estimating
    arena slack; omit slack when ownership is unknown.

    The automatic model is:
        peak = fixed_bytes + batch_size * per_cell_bytes * overhead_factor
        batch_size = (available - fixed_bytes) / (per_cell_bytes * overhead_factor)

    Args:
        n_rows: Number of rows to process.
        fixed_bytes: Costs independent of batch size, such as anchor state.
        per_cell_bytes: Per-row memory footprint.
        overhead_factor: Multiplier for internal temporaries (default: 2.0).
        default_batch: Fallback when GPU probing is unavailable.
        silent: Whether to suppress memory-probe messages.

    Returns:
        Batch size capped at n_rows. An explicit configured size takes priority;
        otherwise automatic estimates use a minimum of 256 before applying the
        row cap, and unavailable probes use default_batch.
    """
    if self.config.batch_size is not None:
        return min(int(self.config.batch_size), n_rows)

    if not tf.config.list_physical_devices('GPU'):
        return min(default_batch, n_rows)

    available_bytes = None

    # ------------------------------------------------------------------
    # 1) Primary: OS-free + our own TF arena slack — correct under both
    #    growth-on (default) and eager-grab modes. (Shared TF-allocator probe.)
    # ------------------------------------------------------------------
    available_bytes = _available_vram_bytes(allocator="tf")
    if available_bytes is not None and not silent:
        self._log(f"  VRAM Info (TF): available≈{available_bytes / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # 2) Fallback: cupy total only (no per-mode info; assume 70 %
    #    headroom as a generic safety margin).
    # ------------------------------------------------------------------
    if available_bytes is None:
        try:
            import cupy as cp
            _, cupy_total = cp.cuda.Device().mem_info
            available_bytes = int(cupy_total * 0.70)
            if not silent:
                self._log(
                    f"  VRAM probe: TF memory info unavailable, "
                    f"using cupy total={cupy_total / 1e9:.1f} GB × 0.70"
                )
        except Exception:
            pass

    # CUDA/TensorFlow sizing; MLX callers use the unified-memory planners.
    if available_bytes is None:
        return min(default_batch, n_rows)

    # ------------------------------------------------------------------
    # 3) Compute batch size (overhead_factor accounts for TF internals)
    # ------------------------------------------------------------------
    if available_bytes <= 0:
        return min(default_batch, n_rows)

    # Subtract fixed anchor costs before sizing the per-cell batch.
    available_for_batch = max(0, int(available_bytes) - int(fixed_bytes))
    if available_for_batch <= 0:
        return min(default_batch, n_rows)

    effective_cost = per_cell_bytes * overhead_factor
    if effective_cost <= 0:
        return min(default_batch, n_rows)

    optimal_batch = max(256, int(available_for_batch / effective_cost))
    return min(optimal_batch, n_rows)

# ---------------------------------------------------------------------------
# HECTOR per-cell abnormality helpers
# ---------------------------------------------------------------------------
_ATYPICAL_FEATURE_NAMES: Tuple[str, ...] = (
    "adherence",
    "jsd",
    "predicted_node_similarity",
    "predicted_class_similarity_margin",
)


def _hybrid_compute_per_cell_features(
    self,
    *,
    cell_embeddings: np.ndarray,
    node_embeddings: np.ndarray,
    predictions: np.ndarray,
    pre_grit_score_matrix: np.ndarray,
    active_indices: List[int],
    tree_edges: List[Tuple[int, int]],
    keep_mask: np.ndarray,
    temperature: float,
    top_k_neighbors: int,
    class_relative_adherence: bool,
    class_relative_blend_weight: float,
    node_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 4-D geometry feature stack.

    Returns:
        ``(features, top1_idx, top2_idx, original_adherence,
        class_relative_adherence)`` where ``features`` is shape
        ``(n_cells, 4)`` with the order in
        ``_ATYPICAL_FEATURE_NAMES``: adherence, jsd,
        predicted_node_similarity, predicted_class_similarity_margin.
        Column 3 is ``sim_top1 - sim_top2`` (the cosine margin between
        the cell's top-1 and top-2 predicted class nodes), naturally
        bounded in ``[-2, +2]``. Cells outside
        ``keep_mask`` (no valid trajectory context) are filled with
        NaN. The ``top1_idx`` and ``top2_idx`` arrays are returned
        for diagnostics; both contain integer indices into
        ``node_names``. ``original_adherence`` is the raw (un-blended)
        adherence; ``class_relative_adherence`` is the class-z
        sigmoid-rescaled adherence prior to blending — NaN where
        class-relative adherence was not applied (disabled, single
        class, or cell outside ``keep_mask``).
    """
    from .trajectory_support import _score_cells_on_active_graph

    n_cells = int(cell_embeddings.shape[0])
    n_features = len(_ATYPICAL_FEATURE_NAMES)
    features = np.full((n_cells, n_features), np.nan, dtype=np.float64)
    top1_idx = np.asarray(predictions, dtype=np.int64).copy()
    top2_idx = np.full(n_cells, -1, dtype=np.int64)
    original_adherence = np.full(n_cells, np.nan, dtype=np.float64)
    class_relative_adherence_arr = np.full(n_cells, np.nan, dtype=np.float64)

    n_kept = int(np.sum(keep_mask))
    if n_kept == 0:
        return (
            features,
            top1_idx,
            top2_idx,
            original_adherence,
            class_relative_adherence_arr,
        )

    kept_embeddings = np.asarray(cell_embeddings[keep_mask], dtype=np.float32)
    kept_pre_grit = np.asarray(pre_grit_score_matrix[keep_mask], dtype=np.float64)
    kept_predictions = top1_idx[keep_mask]

    # Single shared call — gives us adherence, JSD (predictive_divergence),
    # weights matrix and unmasked similarities at once.
    score_result = _score_cells_on_active_graph(
        cell_embeddings=kept_embeddings,
        node_embeddings=node_embeddings,
        active_indices=active_indices,
        tree_edges=tree_edges,
        temperature=temperature,
        top_k_neighbors=top_k_neighbors,
        predictive_scores=kept_pre_grit,
        predicted_classes=kept_predictions,
        class_names=node_names,
        class_relative_adherence=class_relative_adherence,
        class_relative_blend_weight=class_relative_blend_weight,
    )
    adherence_kept = np.asarray(
        score_result["adherence_scores"], dtype=np.float64
    )
    jsd_kept = np.asarray(
        score_result["predictive_divergence"], dtype=np.float64
    )
    original_adherence_kept = np.asarray(
        score_result["original_adherence_scores"], dtype=np.float64
    )
    class_relative_kept = np.asarray(
        score_result["class_relative_scores"], dtype=np.float64
    )

    # Top-1 / top-2 indices from the row-normalized pre-GRIT scores.
    # We need them only as indices into the node-embedding matrix to
    # compute the cell-to-predicted-node similarity and the cosine
    # margin between top-1 and top-2 predictions.
    row_sums = kept_pre_grit.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    prob_full = kept_pre_grit / row_sums
    if prob_full.shape[1] >= 2:
        part = np.argpartition(-prob_full, 1, axis=1)[:, :2]
        row_arange = np.arange(prob_full.shape[0])
        top_two_probs = prob_full[row_arange[:, None], part]
        order = np.argsort(-top_two_probs, axis=1)
        top1_local = part[row_arange, order[:, 0]]
        top2_local = part[row_arange, order[:, 1]]
    else:
        top1_local = np.zeros(prob_full.shape[0], dtype=np.int64)
        top2_local = np.zeros(prob_full.shape[0], dtype=np.int64)

    # Geometric similarity to top-1 and top-2 class node embeddings (cosine).
    node_embeddings_arr = np.asarray(node_embeddings, dtype=np.float64)
    node_norms = np.linalg.norm(node_embeddings_arr, axis=1)
    node_norms = np.where(node_norms > 0, node_norms, 1e-12)
    kept_norms = np.linalg.norm(kept_embeddings.astype(np.float64), axis=1)
    kept_norms = np.where(kept_norms > 0, kept_norms, 1e-12)
    kept_emb64 = kept_embeddings.astype(np.float64)
    sim_top1 = np.einsum(
        "ij,ij->i",
        kept_emb64,
        node_embeddings_arr[top1_local],
    ) / (kept_norms * node_norms[top1_local])
    sim_top2 = np.einsum(
        "ij,ij->i",
        kept_emb64,
        node_embeddings_arr[top2_local],
    ) / (kept_norms * node_norms[top2_local])
    # Cosine margin between the two predicted ontology nodes, bounded in [-2, 2].
    sim_margin = sim_top1 - sim_top2

    kept_features = np.column_stack([
        adherence_kept,
        jsd_kept,
        sim_top1,
        sim_margin,
    ])
    features[keep_mask] = kept_features
    top1_idx[keep_mask] = top1_local
    top2_idx[keep_mask] = top2_local
    original_adherence[keep_mask] = original_adherence_kept
    class_relative_adherence_arr[keep_mask] = class_relative_kept
    return (
        features,
        top1_idx,
        top2_idx,
        original_adherence,
        class_relative_adherence_arr,
    )
def _hybrid_compute_per_cell_abnormality(
    *,
    features: np.ndarray,
) -> np.ndarray:
    """Per-cell abnormality on an absolute [0, 1] scale.

    Maps the 4-D geometry feature stack returned by
    ``_hybrid_compute_per_cell_features`` onto a per-cell
    abnormality score in ``[0, 1]`` (higher = more abnormal), provided the
    inputs satisfy the feature ranges below. The transform uses no fitted
    parameters, but upstream feature computation can depend on the dataset
    and active ontology context.

    Per-feature transforms (each in ``[0, 1]``, higher = more abnormal):
        adh_bad    = 1 - adherence                                  # adherence ∈ [0, 1]
        jsd_bad    = jsd                                            # jsd       ∈ [0, 1] (already)
        node_bad   = (1 - predicted_node_similarity) / 2            # node_sim  ∈ [-1, +1]
        margin_bad = clip((1 - sim_margin) / 2, 0, 1)               # sim_margin ∈ [-2, +2]

    ``abnormality = (adh_bad + jsd_bad + node_bad + margin_bad) / 4``

    Args:
        features: ``(n_cells, 4)`` float64 array with the column order
            in ``_ATYPICAL_FEATURE_NAMES``. NaN rows yield
            NaN abnormality.

    Returns:
        ``(n_cells,)`` float64 abnormality score in ``[0, 1]``, with
        NaN for any cell whose feature row contains NaN.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError(
            f"features must be (n_cells, 4); got shape {features.shape}"
        )
    adherence = features[:, 0]
    jsd = features[:, 1]
    node_sim = features[:, 2]
    margin = features[:, 3]

    adh_bad = 1.0 - adherence
    jsd_bad = jsd
    node_bad = (1.0 - node_sim) / 2.0
    margin_bad = np.clip((1.0 - margin) / 2.0, 0.0, 1.0)

    return (adh_bad + jsd_bad + node_bad + margin_bad) / 4.0


# ---------------------------------------------------------------------------
# Cell QC filter
# ---------------------------------------------------------------------------

_qc_logger = logging.getLogger(__name__)

HUMAN_MITO_GENE_IDS = frozenset({
    "ENSG00000198888",  # MT-ND1
    "ENSG00000198763",  # MT-ND2
    "ENSG00000198804",  # MT-CO1
    "ENSG00000198712",  # MT-CO2
    "ENSG00000228253",  # MT-ATP8
    "ENSG00000198899",  # MT-ATP6
    "ENSG00000198938",  # MT-CO3
    "ENSG00000198840",  # MT-ND3
    "ENSG00000212907",  # MT-ND4L
    "ENSG00000198886",  # MT-ND4
    "ENSG00000198786",  # MT-ND5
    "ENSG00000198695",  # MT-ND6
    "ENSG00000198727",  # MT-CYB
})

MOUSE_MITO_GENE_IDS = frozenset({
    "ENSMUSG00000064341",  # mt-Nd1
    "ENSMUSG00000064345",  # mt-Nd2
    "ENSMUSG00000064351",  # mt-Co1
    "ENSMUSG00000064354",  # mt-Co2
    "ENSMUSG00000064356",  # mt-Atp8
    "ENSMUSG00000064357",  # mt-Atp6
    "ENSMUSG00000064358",  # mt-Co3
    "ENSMUSG00000064360",  # mt-Nd3
    "ENSMUSG00000065947",  # mt-Nd4l
    "ENSMUSG00000064363",  # mt-Nd4
    "ENSMUSG00000064367",  # mt-Nd5
    "ENSMUSG00000064368",  # mt-Nd6
    "ENSMUSG00000064370",  # mt-Cytb
})


def classify_expression_matrix(X, max_nonzero_samples: int = 10000) -> str:
    """Classify an expression matrix from a small non-zero sample.

    Separates three cases for HECTOR's preprocessing/QC paths: integer-like
    raw counts, non-negative log1p-like values, and everything else. The check
    runs on non-zero values only so sparse single-cell matrices are not
    dominated by zeros.

    Args:
        X: Expression matrix to inspect. Dense or sparse.
        max_nonzero_samples: Maximum number of non-zero values to sample.

    Returns:
        One of ``"raw_counts"``, ``"log1p_like"``, or ``"unsupported"``.
    """
    from scipy import sparse

    # Sample nonzero values before converting to float64 to avoid
    # a full-matrix temporary.
    if sparse.issparse(X):
        raw_values = X.data
    else:
        arr = np.asarray(X)
        raw_values = arr[arr != 0]

    if raw_values.size == 0:
        return "unsupported"

    if raw_values.size > max_nonzero_samples:
        idx = np.linspace(0, raw_values.size - 1, max_nonzero_samples, dtype=np.int64)
        raw_values = raw_values[idx]

    values = np.asarray(raw_values, dtype=np.float64)

    if not np.isfinite(values).all():
        return "unsupported"
    if np.any(values < 0):
        return "unsupported"

    integer_like_fraction = np.mean(np.abs(values - np.rint(values)) <= 1e-6)
    small_positive_fraction = np.mean((values > 0.0) & (values < 1.0 - 1e-6))

    if integer_like_fraction >= 0.98 and small_positive_fraction <= 0.001:
        return "raw_counts"

    if (
        integer_like_fraction < 0.8
        and small_positive_fraction >= 0.05
        and values.max() <= 10.0
    ):
        return "log1p_like"

    return "unsupported"


def resolve_expression_source(adata, *, allow_log1p: bool = True, logger=None,
                              return_candidates: bool = False) -> dict:
    """Select the AnnData slot that holds raw counts.

    Candidates are considered in fixed priority order::

        adata.layers["counts"] -> adata.raw.X -> adata.X

    Each present candidate is classified; the FIRST that classifies as
    ``"raw_counts"`` is selected. If none are raw and ``allow_log1p`` is
    True, the first ``"log1p_like"`` candidate is selected with
    ``use_backcalc=True`` (and a warning is emitted via ``logger`` if given).
    If nothing usable is found, a ``ValueError`` naming the checked slots is
    raised. The selected matrix and its matching ``var`` always travel
    together so downstream gene-axis operations stay consistent.

    Args:
        adata: AnnData to resolve.
        allow_log1p: When False (QC/doublet callers), refuse anything that is
            not true raw counts instead of falling back to a log1p recovery
            path.
        logger: Optional one-arg callable for the back-calculation warning.
        return_candidates: When True, also return every present slot (with its
            byte size) under the ``"candidates"`` key -- used by the preflight
            memory guard to flag loaded-but-unused expression copies.

    Returns:
        dict with keys ``name``, ``matrix``, ``var``, ``matrix_type``,
        ``use_backcalc``. When return_candidates=True, also 'candidates':
        list of {name, matrix, var, matrix_type, nbytes} for every present slot.
    """
    candidates = []
    layers = getattr(adata, "layers", None)
    if layers is not None and "counts" in layers:
        candidates.append(
            {"name": "adata.layers['counts']", "matrix": adata.layers["counts"], "var": adata.var}
        )
    raw = getattr(adata, "raw", None)
    if raw is not None:
        candidates.append(
            {"name": "adata.raw.X", "matrix": raw.X, "var": raw.var}
        )
    X = getattr(adata, "X", None)
    if X is not None:
        candidates.append({"name": "adata.X", "matrix": X, "var": adata.var})

    for candidate in candidates:
        candidate["matrix_type"] = classify_expression_matrix(candidate["matrix"])

    use_backcalc = False
    selected = next((c for c in candidates if c["matrix_type"] == "raw_counts"), None)

    if selected is None and allow_log1p:
        selected = next((c for c in candidates if c["matrix_type"] == "log1p_like"), None)
        if selected is not None:
            use_backcalc = True
            if logger is not None:
                logger(
                    f" ℹ️ HECTOR requires raw counts expression data. "
                    f"{selected['name']} does not look like raw counts and "
                    f"prediction will not be accurate."
                )

    if selected is None:
        raise ValueError(
            " ⚠️ HECTOR requires raw counts expression data, but none of "
            "adata.layers['counts'], adata.raw.X, or adata.X matches those patterns."
        )

    resolved = dict(selected)
    resolved["use_backcalc"] = use_backcalc
    if return_candidates:
        # Byte size of each present slot (copy-free: read array attributes only).
        def _nbytes(m):
            import scipy.sparse as _sp
            if _sp.issparse(m):
                return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)
            return int(np.asarray(m).nbytes)

        resolved["candidates"] = [
            {
                "name": c["name"],
                "matrix": c["matrix"],
                "var": c["var"],
                "matrix_type": c["matrix_type"],
                "nbytes": _nbytes(c["matrix"]),
            }
            for c in candidates
        ]
    return resolved


def filter_cells(
    adata,
    min_counts: int = 500,
    min_genes: int = 200,
    max_mito_pct: float = 20.0,
    subset: bool = True,
    *,
    species: Optional[str] = None,
    mito_gene_ids=None,
) -> None:
    """Filter low-quality cells from an AnnData object in-place.

    Computes per-cell QC metrics (total counts, detected genes, mitochondrial
    percentage) and either removes cells that fail any threshold or marks
    them with a pass/fail flag.

    Args:
        adata: AnnData with raw count expression data.
        min_counts: Minimum total UMI counts per cell.
        min_genes: Minimum number of detected genes per cell.
        max_mito_pct: Maximum mitochondrial count percentage per cell.
        subset: When True (default), drop failing cells in-place; the QC
            metric columns are written but the redundant ``hector_qc_pass``
            flag is not. When False, keep all cells and write
            ``hector_qc_pass`` so downstream code can decide what to do.
        species: Optional canonical or common species name; checked against IDs.
        mito_gene_ids: Optional mitochondrial Ensembl IDs for the input annotation.
    """
    from scipy import sparse as sp

    # --- warn if QC columns already exist -----------------------------------
    qc_cols = ["nCount_RNA", "nFeature_RNA", "percent_mt", "hector_qc_pass"]
    existing = [c for c in qc_cols if c in adata.obs.columns]
    if existing:
        warnings.warn(
            f"QC columns {existing} already exist in adata.obs and will be "
            f"overwritten.",
            UserWarning,
            stacklevel=2,
        )

    # --- resolve raw-count expression source --------------------------------
    resolved = resolve_expression_source(adata, allow_log1p=False)
    X = resolved["matrix"]
    _qc_logger.info("filter_cells: using raw counts from %s", resolved["name"])

    # --- auto-detect gene IDs on the resolved matrix's var axis --------------
    gene_ids = _find_ensembl_id_array_in_var(resolved["var"])
    if gene_ids is None:
        raise ValueError(
            "Could not find supported Ensembl gene IDs in adata.var index or columns."
        )

    # --- auto-detect species ------------------------------------------------
    species = _resolve_species(gene_ids, species)
    gene_ids = [_strip_ensembl_version(str(g)) for g in gene_ids]
    annotated_mito = None
    if mito_gene_ids is None:
        var = resolved['var']
        for key in ('mt', 'mitochondrial', 'is_mitochondrial'):
            if key in var and pd.api.types.is_bool_dtype(var[key]) and not var[key].isna().any():
                annotated_mito = var[key].to_numpy(dtype=bool)
                break
        if annotated_mito is None:
            for key in ('chromosome', 'chrom', 'seqname'):
                if key in var:
                    chrom = var[key].astype(str).str.upper().str.removeprefix('CHR')
                    annotated_mito = chrom.isin(['M', 'MT']).to_numpy()
                    break
    if mito_gene_ids is not None:
        mito_set = {_strip_ensembl_version(str(g)) for g in mito_gene_ids}
    elif annotated_mito is not None:
        mito_set = {gene for gene, flag in zip(gene_ids, annotated_mito) if flag}
    else:
        import json
        meta_path = resources.files('hector_pro').joinpath('resources', 'gene_mappings', 'meta.json')
        with meta_path.open('r') as handle:
            mito_set = set(json.load(handle)['mitochondrial_gene_ids'].get(species, []))
        if species == 'homo_sapiens':
            mito_set.update(HUMAN_MITO_GENE_IDS)
        elif species == 'mus_musculus':
            mito_set.update(MOUSE_MITO_GENE_IDS)
        if not mito_set:
            raise ValueError(f'No mitochondrial reference for {species}; supply a boolean '
                             "var['mt'], a chromosome column, or mito_gene_ids.")

    _qc_logger.info("Detected species: %s", species)

    # --- compute per-cell metrics -------------------------------------------
    if sp.issparse(X):
        n_counts = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
        n_genes = np.asarray((X != 0).sum(axis=1)).ravel().astype(np.int64)
    else:
        X_dense = np.asarray(X, dtype=np.float64)
        n_counts = X_dense.sum(axis=1)
        n_genes = (X_dense != 0).sum(axis=1).astype(np.int64)

    # mito percentage
    mito_mask = np.array([g in mito_set for g in gene_ids], dtype=bool)
    n_mito_found = int(mito_mask.sum())

    if n_mito_found == 0:
        _qc_logger.warning(
            "No mitochondrial genes from the %s reference set were found in "
            "the data. Mitochondrial filtering will be skipped (percent_mt "
            "set to 0).",
            species,
        )
        mito_counts = np.zeros(adata.n_obs, dtype=np.float64)
    else:
        if sp.issparse(X):
            mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel().astype(np.float64)
        else:
            mito_counts = np.asarray(X, dtype=np.float64)[:, mito_mask].sum(axis=1)

    with np.errstate(invalid="ignore"):
        pct_mt = np.where(n_counts > 0, mito_counts / n_counts * 100.0, 0.0)

    # --- write QC columns ---------------------------------------------------
    adata.obs["nCount_RNA"] = n_counts
    adata.obs["nFeature_RNA"] = n_genes
    adata.obs["percent_mt"] = pct_mt

    # --- build pass/fail mask -----------------------------------------------
    pass_counts = n_counts >= min_counts
    pass_genes = n_genes >= min_genes
    pass_mito = pct_mt <= max_mito_pct
    qc_pass = pass_counts & pass_genes & pass_mito

    # --- log summary --------------------------------------------------------
    n_total = len(qc_pass)
    n_fail = int((~qc_pass).sum())
    n_fail_counts = int((~pass_counts).sum())
    n_fail_genes = int((~pass_genes).sum())
    n_fail_mito = int((~pass_mito).sum())

    action_word = "removed" if subset else "flagged"
    if n_fail == 0:
        _qc_logger.info(
            "Cell QC: all %d cells passed (min_counts=%d, min_genes=%d, "
            "max_mito_pct=%.1f).",
            n_total, min_counts, min_genes, max_mito_pct,
        )
    else:
        _qc_logger.info(
            "Cell QC: %s %d / %d cells (%d low counts, %d low genes, "
            "%d high mito%%).",
            action_word, n_fail, n_total, n_fail_counts, n_fail_genes, n_fail_mito,
        )

    if n_fail == n_total:
        if subset:
            _qc_logger.warning(
                "All cells were filtered out (0 cells remain). Downstream "
                "predict() will receive an empty AnnData."
            )
        else:
            _qc_logger.warning(
                "All cells failed QC (hector_qc_pass is False for every cell)."
            )

    # --- subset or flag -----------------------------------------------------
    if subset:
        adata._inplace_subset_obs(qc_pass)
    else:
        adata.obs["hector_qc_pass"] = qc_pass


# =============================================================================
# Doublet detection helpers
# =============================================================================

def compute_cxds_score(*, count_matrix, ntop: int = 500):
    """Compute co-expression-based doublet score (cxds, Bais & Kostka 2020).

    Scores each cell by how many mutually-exclusive gene pairs it
    co-expresses.  Doublets from different cell types co-express markers
    that are normally on in different subpopulations, yielding a high score.

    Returns scores as a float64 array of shape (n_cells,).
    """
    import scipy.sparse
    from scipy.stats import binom

    if scipy.sparse.issparse(count_matrix):
        B = (count_matrix > 0).astype(np.float64)
    else:
        B = (np.asarray(count_matrix) > 0).astype(np.float64)

    n_cells = B.shape[0]

    if scipy.sparse.issparse(B):
        p = np.asarray(B.mean(axis=0)).flatten()
    else:
        p = B.mean(axis=0)

    binom_var = p * (1 - p)
    n_select = min(ntop, len(binom_var))
    top_idx = np.argsort(binom_var)[::-1][:n_select]

    if scipy.sparse.issparse(B):
        B = B[:, top_idx].toarray()
    else:
        B = np.asarray(B[:, top_idx])

    p = B.mean(axis=0)

    prb = np.outer(p, 1 - p) + np.outer(1 - p, p)
    prb = np.clip(prb, 1e-15, 1 - 1e-15)

    BtB = B.T @ B
    B_col_sums = B.sum(axis=0)
    obs = (B_col_sums[:, None] - BtB) + (B_col_sums[None, :] - BtB)

    S = binom.logsf(obs.astype(int) - 1, n_cells, prb)
    np.fill_diagonal(S, 0.0)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=-700.0)

    BS = B @ S
    return (-np.sum(BS * B, axis=1)).astype(np.float64)


def compute_doublet_scores(
    *,
    count_matrix,
    random_seed: int = 0,
) -> tuple:
    """Compute doublet scores via count-space synthetics, density ratio, and cxds.

    Three complementary signals, combined by weighted rank averaging:
      - pANN (proportion of artificial nearest neighbors, up to k=100)
      - density ratio (synthetic density / singlet density, up to k_dr=50)
      - cxds (binary co-expression of mutually-exclusive gene pairs)

    pANN and density ratio are extracted from a single up-to-k=300 combined-space
    kNN query.  cxds is computed independently on the binarized count matrix.

    Returns (doublet_scores, pann_scores, density_ratio_scores, cxds_scores)
    as float64 arrays of shape (n_cells,).
    """
    import scipy.sparse
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import rankdata

    _K_PANN = 100
    _K_DR = 50
    _K_COMBINED = 300
    _N_SYNTHETIC_RATIO = 2.0
    _N_PCS = 50
    _N_TOP_GENES = 3000
    _W_PANN = 0.40
    _W_DR = 0.40
    _W_CXDS = 0.20

    n_cells = count_matrix.shape[0]
    n_synthetic = max(1, int(n_cells * _N_SYNTHETIC_RATIO))
    rng = np.random.default_rng(random_seed)

    # --- cxds (independent of synthetics) ---
    cxds_scores = compute_cxds_score(count_matrix=count_matrix)

    # --- Generate synthetic doublets by summing raw counts ---
    idx_a = rng.integers(0, n_cells, size=n_synthetic)
    idx_b = rng.integers(0, n_cells, size=n_synthetic)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % n_cells

    if scipy.sparse.issparse(count_matrix):
        synthetic_counts = count_matrix[idx_a] + count_matrix[idx_b]
        combined_counts = scipy.sparse.vstack([count_matrix, synthetic_counts])
    else:
        count_arr = np.asarray(count_matrix)
        synthetic_counts = count_arr[idx_a] + count_arr[idx_b]
        combined_counts = np.vstack([count_arr, synthetic_counts])

    # --- Joint normalization and PCA ---
    import scanpy as sc
    import anndata as ad

    adata_tmp = ad.AnnData(X=combined_counts)
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    n_hvg = min(_N_TOP_GENES, adata_tmp.shape[1])
    if n_hvg < adata_tmp.shape[1]:
        sc.pp.highly_variable_genes(adata_tmp, n_top_genes=n_hvg)
        adata_tmp = adata_tmp[:, adata_tmp.var['highly_variable']].copy()
    if scipy.sparse.issparse(adata_tmp.X):
        adata_tmp.X = adata_tmp.X.toarray()
    sc.pp.scale(adata_tmp, max_value=10)
    n_comps = min(_N_PCS, adata_tmp.shape[1] - 1, adata_tmp.shape[0] - 1)
    sc.tl.pca(adata_tmp, n_comps=n_comps, random_state=random_seed)
    combined_emb = adata_tmp.obsm['X_pca']
    real_emb = combined_emb[:n_cells]
    del adata_tmp

    # --- Single kNN at k=_K_COMBINED for both pANN and density ratio ---
    k_use = min(_K_COMBINED, combined_emb.shape[0] - 1)
    nn = NearestNeighbors(
        n_neighbors=k_use + 1, algorithm='auto', metric='euclidean'
    )
    nn.fit(combined_emb)
    _, indices = nn.kneighbors(real_emb)
    neighbors = indices[:, 1:]

    # pANN from first _K_PANN neighbors
    k_pann = min(_K_PANN, neighbors.shape[1])
    pann_scores = (neighbors[:, :k_pann] >= n_cells).sum(axis=1).astype(np.float64) / k_pann

    # Density ratio: count synthetics before the _K_DR-th real neighbor
    k_dr = min(_K_DR, n_cells - 1)
    is_real = neighbors < n_cells
    real_cumsum = np.cumsum(is_real, axis=1)
    hit = (real_cumsum == k_dr)
    reached = hit.any(axis=1)
    col_idx = np.argmax(hit, axis=1)
    synth_count = (col_idx + 1).astype(np.float64) - k_dr
    if not reached.all():
        synth_count[~reached] = np.sum(
            ~is_real[~reached], axis=1
        ).astype(np.float64)
    dr_scores = synth_count * n_cells / (n_synthetic * k_dr)

    # --- Weighted rank combination ---
    rank_pann = rankdata(pann_scores) / n_cells
    rank_dr = rankdata(dr_scores) / n_cells
    rank_cxds = rankdata(cxds_scores) / n_cells
    doublet_scores = _W_PANN * rank_pann + _W_DR * rank_dr + _W_CXDS * rank_cxds

    return doublet_scores, pann_scores, dr_scores, cxds_scores


def detect_doublets(
    adata,
    expected_doublet_rate: float = 0.08,
    batch_key: str = None,
) -> None:
    """Flag likely doublet cells using pANN, density ratio, and cxds.

    Generates synthetic doublets by summing raw count vectors of random
    cell pairs within each batch, projects real and synthetic cells
    jointly through normalization and PCA, then scores each cell by a
    weighted combination of three signals: pANN (proportion of artificial
    nearest neighbors), density ratio (local synthetic-to-singlet density),
    and cxds (binary co-expression of mutually-exclusive gene pairs).
    Writes 'doublet_score' and 'is_doublet' to adata.obs.

    When batch_key is provided, runs scoring independently per batch and
    applies the doublet rate threshold per batch, since doublets can only
    form within a single capture (e.g., one 10x lane).

    Requires raw counts in adata.layers['counts'], adata.raw.X, or adata.X
    (resolved in that priority order).
    """
    import warnings

    if expected_doublet_rate <= 0 or expected_doublet_rate >= 1:
        raise ValueError("expected_doublet_rate must be between 0 and 1 (exclusive).")
    if batch_key is not None and batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs.")

    resolved = resolve_expression_source(adata, allow_log1p=False)
    count_matrix = resolved["matrix"]
    _qc_logger.info("detect_doublets: using raw counts from %s", resolved["name"])

    n_cells = adata.shape[0]
    _MIN_CELLS = 302

    if batch_key is None:
        doublet_score, _, _, _ = compute_doublet_scores(
            count_matrix=count_matrix,
            random_seed=0,
        )
        threshold = np.quantile(doublet_score, 1.0 - expected_doublet_rate)
        is_doublet = doublet_score >= threshold
    else:
        batches = adata.obs[batch_key].unique()
        doublet_score = np.full(n_cells, np.nan, dtype=np.float64)
        is_doublet = np.zeros(n_cells, dtype=bool)

        for batch in batches:
            mask = (adata.obs[batch_key] == batch).values
            batch_idx = np.where(mask)[0]
            n_batch = len(batch_idx)

            if n_batch < _MIN_CELLS:
                warnings.warn(
                    f"Batch '{batch}' has only {n_batch} cells "
                    f"(need at least {_MIN_CELLS}), "
                    f"skipping doublet detection for this batch.",
                    UserWarning, stacklevel=2,
                )
                doublet_score[batch_idx] = 0.0
                continue

            batch_scores, _, _, _ = compute_doublet_scores(
                count_matrix=count_matrix[batch_idx],
                random_seed=0,
            )
            doublet_score[batch_idx] = batch_scores
            threshold = np.quantile(batch_scores, 1.0 - expected_doublet_rate)
            is_doublet[batch_idx] = batch_scores >= threshold

    adata.obs['doublet_score'] = doublet_score
    adata.obs['is_doublet'] = is_doublet


# ============================================================================
# Gene-ID format detection, symbol→Ensembl mapping, reference coverage
# ----------------------------------------------------------------------------
# Standalone symbol and coverage helpers use the packaged six-species reference.
# Model inference instead uses the native gene mappings saved in its checkpoint.
# ============================================================================

Species = Literal[
    'human', 'mouse', 'macaque', 'marmoset', 'chimpanzee', 'zebrafish',
    'homo_sapiens', 'mus_musculus', 'macaca_mulatta', 'callithrix_jacchus',
    'pan_troglodytes', 'danio_rerio',
]

_SPECIES_PARQUET_PREFIX = {
    'mouse': 'mus_musculus', 'human': 'homo_sapiens',
    'macaque': 'macaca_mulatta', 'marmoset': 'callithrix_jacchus',
    'chimpanzee': 'pan_troglodytes', 'zebrafish': 'danio_rerio',
}
_SPECIES_ENSEMBL_PREFIX = {
    'mouse': 'ENSMUSG', 'human': 'ENSG', 'macaque': 'ENSMMUG',
    'marmoset': 'ENSCJAG', 'chimpanzee': 'ENSPTRG', 'zebrafish': 'ENSDARG',
}


def _canonical_species(species: str) -> str:
    value = str(species).strip().lower().replace(' ', '_')
    aliases = {'rhesus_macaque': 'macaque', 'rhesus': 'macaque',
               'common_marmoset': 'marmoset', 'chimp': 'chimpanzee'}
    value = aliases.get(value, value)
    canonical = _SPECIES_PARQUET_PREFIX.get(value, value)
    if canonical not in _SPECIES_PARQUET_PREFIX.values():
        raise ValueError(f'Unsupported species {species!r}; choose from '
                         f'{", ".join(_SPECIES_PARQUET_PREFIX)}.')
    return canonical


def _resolve_species(ids, species=None) -> str:
    """Resolve one input species, rejecting mixed IDs and conflicting overrides."""
    ids = tuple(ids)
    detected = {sp for sp, prefix in _SPECIES_ENSEMBL_PREFIX.items()
                if any(str(g).startswith(prefix) for g in ids)}
    if len(detected) > 1:
        raise ValueError('Mixed-species gene IDs found. Predict each species separately.')
    if species is None:
        if not detected:
            raise ValueError('Cannot infer species from gene IDs; provide species explicitly '
                             'and map gene symbols to Ensembl IDs before prediction.')
        return _SPECIES_PARQUET_PREFIX[detected.pop()]
    canonical = _canonical_species(species)
    if detected and _SPECIES_PARQUET_PREFIX[next(iter(detected))] != canonical:
        raise ValueError(f'Gene IDs conflict with the requested species {species!r}.')
    return canonical


@dataclass
class CoverageReport:
    n_found: int
    n_required: int
    pct_found: float
    species: Species
    missing_sample: list = field(default_factory=list)


_ENSEMBL_PATTERN = re.compile(r"^ENS(?:MUS|MMU|CJA|PTR|DAR)?G\d+(?:\.\d+)?$")


def _strip_ensembl_version(gene_id: str) -> str:
    """ENSG00000123456.7 → ENSG00000123456. Idempotent on plain IDs."""
    if not isinstance(gene_id, str):
        return str(gene_id)
    dot = gene_id.find(".")
    return gene_id[:dot] if dot > 0 else gene_id


def _ensembl_fraction(series) -> float:
    """Fraction of non-empty entries in *series* that look like Ensembl gene IDs."""
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""]
    if len(sample) == 0:
        return 0.0
    return float(sample.str.contains(_ENSEMBL_PATTERN, regex=True).sum()) / len(sample)


def _find_ensembl_id_array_in_var(var) -> Optional[np.ndarray]:
    """Locate an Ensembl-ID-bearing array in a var DataFrame — any column or the index."""
    best_frac, best_arr = 0.0, None

    for col in var.columns:
        frac = _ensembl_fraction(var[col])
        if frac > best_frac:
            best_frac = frac
            best_arr = var[col].astype(str).values

    idx_frac = _ensembl_fraction(pd.Series(var.index))
    if idx_frac > best_frac:
        best_frac = idx_frac
        best_arr = np.asarray(var.index.astype(str))

    return best_arr if best_frac > 0.5 else None


def _find_ensembl_id_array(adata) -> Optional[np.ndarray]:
    """Locate an Ensembl-ID-bearing array on ``adata`` — any var column or the index.

    Mirrors the detection logic used by :func:`filter_cells` so all call sites
    agree on what counts as "Ensembl IDs are present". Returns ``None`` if no
    column matches.
    """
    return _find_ensembl_id_array_in_var(adata.var)


def _species_from_ensembl_ids(ids) -> Species:
    canonical = _resolve_species(ids)
    return next(sp for sp, name in _SPECIES_PARQUET_PREFIX.items() if name == canonical)


def _gene_mapping_resource_path():
    return resources.files("hector_pro").joinpath(
        "resources", "gene_mappings", "gene_mapping.parquet"
    )


@functools.lru_cache(maxsize=1)
def _load_gene_mapping_table() -> pd.DataFrame:
    """Load the six-species auxiliary gene reference (cached)."""
    path = _gene_mapping_resource_path()
    with resources.as_file(path) as p:
        return pd.read_parquet(p)


def _species_symbol_ensembl_pairs(df: pd.DataFrame, species: Species):
    """Yield ``(symbol, ensembl_id)`` pairs for ``species`` where the symbol is non-empty."""
    prefix = _canonical_species(species)
    sym_col, ens_col = f"{prefix}_symbol", f"{prefix}_ensembl"
    alias_col = f'{prefix}_aliases'
    aliases = df[alias_col].astype(str) if alias_col in df else [''] * len(df)
    for ens, sym, alias in zip(df[ens_col].astype(str), df[sym_col].astype(str), aliases):
        if not ens:
            continue
        symbols = {sym} | set(alias.split('|'))
        for name in sorted(symbols - {'', ens}):
            yield name, ens


def _load_reference_genes(species: Species) -> List[str]:
    """Return packaged reference Ensembl IDs ordered by ``column_index``."""
    df = _load_gene_mapping_table()
    ens_col = f"{_canonical_species(species)}_ensembl"
    ref = df[df[ens_col].astype(str) != ""].sort_values("column_index")
    return [str(x) for x in ref[ens_col].tolist()]


@functools.lru_cache(maxsize=1)
def load_gene_vocab():
    """Return the packaged six-species reference vocabulary, ordered by column."""
    path = resources.files("hector_pro").joinpath("resources", "gene_mappings", "gene_vocab.csv")
    with resources.as_file(path) as p:
        with open(p, newline="") as fh:
            rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["column_index"]))
    return rows


_SPECIES_ENSEMBL_COL = {sp: f'{name}_ensembl' for sp, name in _SPECIES_PARQUET_PREFIX.items()}


def build_model_input_vector(adata, species, *, gene_ids=None, gene_columns=None,
                             num_genes=None, input_gene_ids=None):
    """Align raw counts to union columns as CSR, summing duplicate input genes.

    Prediction passes all three mapping arguments from its checkpoint. Without
    them this standalone helper uses the packaged six-species reference.
    """
    from scipy.sparse import csc_matrix, csr_matrix
    canonical = _canonical_species(species)
    if gene_ids is None and gene_columns is None and num_genes is None:
        vocab = load_gene_vocab()
        col_name = f'{canonical}_ensembl'
        pairs = [(r[col_name], int(r['column_index'])) for r in vocab if r[col_name]]
        gene_ids, gene_columns = zip(*pairs)
        num_genes = len(vocab)
    elif gene_ids is None or gene_columns is None or num_genes is None:
        raise ValueError('Provide gene_ids, gene_columns, and num_genes together.')
    if len(gene_ids) != len(gene_columns) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError('Invalid model gene mapping.')
    n_vocab = int(num_genes)
    columns = np.asarray(gene_columns)
    if (columns.ndim != 1 or not np.issubdtype(columns.dtype, np.integer)
            or len(np.unique(columns)) != len(columns)
            or np.any(columns < 0) or np.any(columns >= n_vocab)):
        raise ValueError('Invalid model gene column indices.')
    ens_to_column = dict(zip(gene_ids, columns))
    ids = input_gene_ids if input_gene_ids is not None else adata.var_names
    if len(ids) != adata.n_vars:
        raise ValueError('Input gene IDs must match the expression columns.')
    src_cols, dst_cols = [], []
    for j, gene in enumerate(ids):
        column = ens_to_column.get(_strip_ensembl_version(str(gene).strip()))
        if column is not None:
            src_cols.append(j)
            dst_cols.append(column)
    if not src_cols:
        return csr_matrix((adata.n_obs, n_vocab), dtype=np.float32)
    src_cols = np.asarray(src_cols, dtype=np.int64)
    dst_cols = np.asarray(dst_cols, dtype=np.int64)
    X = adata.X
    Xc = X.tocsc() if issparse(X) else csc_matrix(np.asarray(X, dtype=np.float32))
    X_sel = Xc[:, src_cols]                      # fast column slice on CSC
    # scatter selected columns to their vocab positions: out = X_sel @ P, where
    # P[k, dst_cols[k]] = 1 places selected column k at vocab column dst_cols[k].
    P = csr_matrix(
        (np.ones(src_cols.size, dtype=np.float32),
         (np.arange(src_cols.size), dst_cols)),
        shape=(src_cols.size, n_vocab),
    )
    return (X_sel @ P).tocsr().astype(np.float32, copy=False)


def _compute_coverage(adata, species: Species) -> CoverageReport:
    """Fraction of packaged reference genes present in ``adata``."""
    ref = _load_reference_genes(species)
    ref_set = set(ref)
    ids = _find_ensembl_id_array(adata)
    if ids is None:
        ids = np.asarray(adata.var_names.astype(str))
    present = {_strip_ensembl_version(x) for x in ids}

    found = ref_set & present
    missing = ref_set - present
    n_required = len(ref_set)
    n_found = len(found)
    pct = (100.0 * n_found / n_required) if n_required else 0.0
    sample = sorted(missing)[:20]

    return CoverageReport(
        n_found=n_found,
        n_required=n_required,
        pct_found=pct,
        species=species,
        missing_sample=sample,
    )


def _autodetect_species_from_symbols(adata) -> Species:
    """Infer species only when symbols match exactly one supported species.

    Require an explicit species when no species or multiple species match.
    """
    syms = set(str(s) for s in adata.var_names)
    df = _load_gene_mapping_table()
    hits: Dict[str, int] = {}
    for species in _SPECIES_PARQUET_PREFIX:
        sym_col = f"{_SPECIES_PARQUET_PREFIX[species]}_symbol"
        if sym_col not in df:
            continue
        species_syms = set(df[sym_col].astype(str)) - {""}
        hits[species] = len(syms & species_syms)

    best = max(hits.values(), default=0)
    if best == 0 or sum(count > 0 for count in hits.values()) != 1:
        raise ValueError(
            "Could not unambiguously infer species from gene symbols. "
            "Pass species explicitly (human, mouse, macaque, marmoset, "
            "chimpanzee, or zebrafish)."
        )
    return max(hits, key=hits.get)  # type: ignore[return-value]


_MAP_GENE_SYMBOLS_OUTPUT_COLUMN = "ensembl_id"


def map_gene_symbols(adata, *, species: Optional[Species] = None) -> Dict[str, object]:
    """Make sure ``adata`` has Ensembl gene IDs that HECTOR can find.

    Idempotent one-shot helper:

    * If an Ensembl-ID array is detected in a ``adata.var`` column or the
      index, validates the species and returns without changes.
    * Otherwise, looks up symbols from ``adata.var_names`` in the bundled
      species table and writes Ensembl IDs to ``adata.var['ensembl_id']``.
      Ambiguous or unmapped symbols are left
      as empty strings — downstream prediction treats them as missing (zero-
      filled), so the matrix shape is preserved.
    * Species is auto-detected from the symbols when not specified.

    Returns a small report dict: ``{species, n_input, n_mapped, n_ambiguous,
    n_unmapped}``.
    """
    n_input = adata.n_vars

    # Already Ensembl somewhere? No-op.
    found = _find_ensembl_id_array(adata)
    if found is not None:
        _resolve_species(found, species)
        return {
            "species": _species_from_ensembl_ids(found),
            "n_input": n_input,
            "n_mapped": n_input,
            "n_ambiguous": 0,
            "n_unmapped": 0,
        }

    if species is None:
        species = _autodetect_species_from_symbols(adata)
    _canonical_species(species)

    df = _load_gene_mapping_table()
    by_symbol: Dict[str, set] = {}
    for sym, ens in _species_symbol_ensembl_pairs(df, species):
        by_symbol.setdefault(sym, set()).add(ens)

    out: List[str] = []
    n_mapped = n_ambiguous = n_unmapped = 0
    for sym in adata.var_names:
        hits = by_symbol.get(str(sym))
        if hits is None:
            out.append("")
            n_unmapped += 1
        elif len(hits) > 1:
            out.append("")
            n_ambiguous += 1
        else:
            out.append(next(iter(hits)))
            n_mapped += 1

    adata.var[_MAP_GENE_SYMBOLS_OUTPUT_COLUMN] = out

    if n_unmapped + n_ambiguous > 0:
        warnings.warn(
            f"map_gene_symbols: {n_unmapped + n_ambiguous}/{n_input} symbols "
            f"left as empty ({n_ambiguous} ambiguous, {n_unmapped} unmapped).",
            stacklevel=2,
        )

    return {
        "species": species,
        "n_input": n_input,
        "n_mapped": n_mapped,
        "n_ambiguous": n_ambiguous,
        "n_unmapped": n_unmapped,
    }


# =============================================================================
# Marker-gene statistics
# =============================================================================

def _group_sums(X, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum the rows of X within each integer group code → ``[n_groups, n_genes]``.

    Uses a sparse one-hot matrix multiplication with dense or sparse X.
    """
    from scipy.sparse import csr_matrix
    n = X.shape[0]
    # match the one-hot dtype to X so a float32 attribution matrix isn't upcast
    # to float64 (a full copy) on every grouped matmul
    dt = np.float32 if getattr(X, 'dtype', None) == np.float32 else np.float64
    onehot = csr_matrix((np.ones(n, dtype=dt),
                         (np.arange(n), codes)), shape=(n, n_groups))
    s = onehot.T.dot(X)
    return s.toarray() if issparse(s) else np.asarray(s)


def aggregate_attributions(attr, labels, contrast_clip: float = 100.0) -> Dict[str, dict]:
    """Aggregate per-cell (signed) scores into per-type marker statistics.

    Args:
        attr: ``[n_cells, n_genes]`` per-cell scores (dense or sparse). May be
            signed — positive pushes a cell toward its type, negative away.
        labels: ``[n_cells]`` cell-type label per row.
        contrast_clip: cap for the reported contrast ratio (guards the
            divide-by-~0 blow-ups when a gene is ~absent in the other types).

    Returns:
        ``{type: {'mean', 'spec', 'contrast', 'n'}}`` — each a ``[n_genes]``
        array except ``n``:

        * ``mean`` — signed mean score in the type (direction + magnitude).
        * ``spec`` — ``mean_t - max_other``, the signed difference from the
          highest mean in any other type. This is the ranking metric.
        * ``contrast`` — ``max(mean_t, 0) / max(max_other, 1e-9)``, clipped to
          ``[0, contrast_clip]``. A ratio near 1 indicates similar positive
          attribution inside and outside the type.

    Vectorized: per-type means come from a single grouped matmul and the
    one-vs-rest ``max_other`` from a top-2 pass, so cost is O(n_cells·n_genes)
    rather than O(n_types²·n_genes) — required to permute at scale.
    """
    types, codes = np.unique(np.asarray(labels), return_inverse=True)
    K = len(types)
    counts = np.bincount(codes, minlength=K).astype(np.float64)
    means = _group_sums(attr, codes, K) / np.maximum(counts[:, None], 1.0)  # [K, genes]

    if K == 1:
        max_other = np.zeros_like(means)
    else:
        cols = np.arange(means.shape[1])
        argmax = means.argmax(0)
        max1 = means.max(0)
        tmp = means.copy()
        tmp[argmax, cols] = -np.inf
        max2 = tmp.max(0)
        max_other = np.where(np.arange(K)[:, None] == argmax[None, :],
                             max2[None, :], max1[None, :])

    out: Dict[str, dict] = {}
    for i, t in enumerate(types):
        spec = means[i] - max_other[i]
        pos_t = np.maximum(means[i], 0.0)
        pos_other = np.maximum(max_other[i], 0.0)
        contrast = np.clip(pos_t / np.maximum(pos_other, 1e-9), 0.0, contrast_clip)
        out[str(t)] = {'mean': means[i], 'spec': spec, 'contrast': contrast,
                       'n': int(counts[i])}
    return out


def permutation_confidence(attr: np.ndarray, labels: np.ndarray, score: str = 'mean',
                           n_perm: int = 200, seed: int = 0) -> Dict[str, np.ndarray]:
    """Per-gene confidence as an empirical p-value against shuffled labels.

    For each gene and type, counts how often the per-type ``score`` under
    randomly permuted labels is >= the observed score. A type-specific gene is
    rarely matched by chance (low value); a ubiquitous or flat gene is matched
    often (high value). Smaller is more confident.

    Args:
        attr: ``[n_cells, n_genes]`` per-cell scores (attribution or counts).
        labels: ``[n_cells]`` cell-type labels.
        score: which aggregate to test, ``'mean'`` or ``'contrast'``.
        n_perm: number of label shuffles.
        seed: Seed for reproducible label permutations.

    Returns:
        ``{type: [n_genes]}`` empirical p-values in (0, 1].
    """
    rng = np.random.default_rng(seed)
    obs = {t: d[score] for t, d in aggregate_attributions(attr, labels).items()}
    ge = {t: np.zeros_like(v) for t, v in obs.items()}
    for _ in range(n_perm):
        perm = aggregate_attributions(attr, rng.permutation(labels))
        for t in obs:
            ge[t] += (perm[t][score] >= obs[t]).astype(float)
    return {t: (ge[t] + 1.0) / (n_perm + 1.0) for t in obs}


def onevsrest_confidence(X, labels) -> Dict[str, np.ndarray]:
    """One-sided confidence using a normal approximation and Welch standard error.

    Compare scores within each type against the remaining cells using grouped
    sums and sums of squares. Small upper-tail normal probabilities indicate
    elevated scores in the type; lower scores yield values near 1.
    Groups with fewer than two cells return 1.

    Args:
        X: ``[n_cells, n_genes]`` per-cell scores (attribution or counts), dense
            or sparse.
        labels: ``[n_cells]`` cell-type labels.

    Returns:
        ``{type: [n_genes]}`` upper-tail probabilities in [0, 1].
    """
    types, codes = np.unique(np.asarray(labels), return_inverse=True)
    K = len(types)
    n = X.shape[0]
    cnt = np.bincount(codes, minlength=K).astype(np.float64)
    X2 = X.multiply(X) if issparse(X) else np.asarray(X) ** 2
    gsum = _group_sums(X, codes, K)
    gsq = _group_sums(X2, codes, K)
    tot = np.asarray(X.sum(0)).ravel()
    totsq = np.asarray(X2.sum(0)).ravel()
    out: Dict[str, np.ndarray] = {}
    for i, t in enumerate(types):
        n_t, n_r = cnt[i], n - cnt[i]
        if n_t < 2 or n_r < 2:
            out[str(t)] = np.ones(gsum.shape[1])
            continue
        m_t = gsum[i] / n_t
        m_r = (tot - gsum[i]) / n_r
        v_t = np.maximum(gsq[i] / n_t - m_t ** 2, 0.0) * (n_t / (n_t - 1.0))
        v_r = np.maximum((totsq - gsq[i]) / n_r - m_r ** 2, 0.0) * (n_r / (n_r - 1.0))
        se = np.sqrt(v_t / n_t + v_r / n_r) + 1e-12
        tstat = (m_t - m_r) / se
        out[str(t)] = scipy_stats.norm.sf(tstat).astype(np.float64)
    return out


def combine_across_samples(per_sample: Dict[str, Dict[str, np.ndarray]],
                           thresh: float = 0.0) -> Dict[str, np.ndarray]:
    """Cross-sample reproducibility: fraction of samples where a gene's effect
    exceeds ``thresh`` for each type.

    Args:
        per_sample: ``{sample: {type: [n_genes] effect}}`` — a per-sample, per-type
            effect array (e.g. contrast for the model side, level gap for the
            observed side).
        thresh: a gene "holds" in a sample when its effect > ``thresh``.

    Returns:
        ``{type: [n_genes]}`` fraction of samples containing that type in
        which the gene's effect exceeds the threshold.
    """
    samples = list(per_sample)
    types = {t for d in samples for t in per_sample[d]}
    out: Dict[str, np.ndarray] = {}
    for t in types:
        stacks = [per_sample[d][t] for d in samples if t in per_sample[d]]
        if not stacks:
            continue
        out[t] = (np.vstack(stacks) > thresh).mean(0)
    return out


def across_sample_confidence(attr, labels, samples, min_cells: int = 10):
    """One-sided confidence that a gene's attribution is elevated in a type vs
    the rest, tested ACROSS samples (the sample is the unit of replication).

    For each type and each qualifying sample (>= ``min_cells`` cells in the type
    AND >= ``min_cells`` in the rest), compute the within-sample difference
    ``d_s[gene] = mean_attr(type, s) - mean_attr(rest, s)``. Then test
    H0: ``mean_s d_s <= 0`` with a one-sample upper-tail t-test across the
    qualifying samples. Samples, rather than individual cells, are the
    replication units for this test.

    Args:
        attr: ``[n_cells, n_genes]`` per-cell scores (dense or sparse).
        labels: ``[n_cells]`` cell-type label per row.
        samples: ``[n_cells]`` sample/replicate id per row.
        min_cells: per-sample floor applied to BOTH the type group and the rest
            group; a sample is excluded unless it has >= ``min_cells`` cells in each.

    Returns:
        ``({type: [n_genes] p in [0, 1]}, {type: n_qualifying_samples})``. Types
        with fewer than 2 qualifying samples get all-ones (caller falls back to
        the within-sample test).
    """
    attr = attr.toarray() if issparse(attr) else np.asarray(attr)
    labels = np.asarray(labels)
    samples = np.asarray(samples)
    n_genes = attr.shape[1]
    out, nsamp = {}, {}
    uniq_s = np.unique(samples)
    for t in np.unique(labels):
        in_t = labels == t
        diffs = []
        for s in uniq_s:
            in_s = samples == s
            a = in_s & in_t
            b = in_s & (~in_t)
            if int(a.sum()) >= min_cells and int(b.sum()) >= min_cells:
                diffs.append(attr[a].mean(0) - attr[b].mean(0))
        nsamp[str(t)] = len(diffs)
        if len(diffs) < 2:
            out[str(t)] = np.ones(n_genes)
            continue
        D = np.vstack(diffs)
        _, p = scipy_stats.ttest_1samp(D, 0.0, axis=0, alternative="greater")
        out[str(t)] = np.where(np.isfinite(p), p, 1.0)
    return out, nsamp


def _benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (monotone, clipped to 1)."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def assemble_marker_table(agg, conf, conf_basis, rep, gene_ids, gene_names,
                          name_map):
    """Build a marker table ranked by ``spec`` within each type.

    Confidence breaks ties. ``rep`` (cross-sample reproducibility) may be None.

    Args:
        agg: dict mapping cell-type label to the dict returned by
            ``aggregate_attributions`` — keys ``"mean"``, ``"spec"``,
            ``"contrast"``.
        conf: dict mapping cell-type label to ``[n_genes]`` p-values from
            ``onevsrest_confidence`` or ``across_sample_confidence``.
        conf_basis: dict mapping cell-type label to a human-readable string
            describing which confidence test was used.
        rep: dict mapping cell-type label to ``[n_genes]`` reproducibility
            scores, or ``None`` if not available.
        gene_ids: list of gene identifiers (length ``n_genes``).
        gene_names: list of human-readable gene names (length ``n_genes``).
        name_map: dict mapping cell-type label to a display name.

    Returns:
        A ``pandas.DataFrame`` with one row per (cell_type, gene) combination,
        sorted by ``spec`` descending within each type, with a 1-based ``rank``
        column.
    """
    n = len(gene_ids)
    blocks = []
    for t in agg:
        df = pd.DataFrame({
            "cell_type": t,
            "cell_type_name": name_map.get(t, t),
            "gene_id": gene_ids,
            "gene_name": gene_names,
            "attribution_score": agg[t]["mean"],
            "spec": agg[t]["spec"],
            "contrast_score": agg[t]["contrast"],
            "confidence": conf[t],
            "confidence_basis": conf_basis.get(t, "within_sample"),
            "sample_reproducibility": (
                rep[t] if (rep is not None and t in rep) else np.full(n, np.nan)),
        })
        df = df.sort_values(["spec", "confidence"],
                            ascending=[False, True]).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        blocks.append(df)
    cols = ["cell_type", "cell_type_name", "gene_id", "gene_name",
            "attribution_score", "spec", "contrast_score", "confidence",
            "confidence_basis", "sample_reproducibility", "rank"]
    return pd.concat(blocks, ignore_index=True)[cols]


def contain_markers(df, spec_min: float = 0.0, top_n=100, fdr_alpha=None,
                    return_all: bool = False):
    """Filter a ranked marker table by specificity, optional FDR, and rank.

    ``return_all`` returns ``df`` with a reset index. Otherwise keep
    ``spec > spec_min``, then optionally keep BH-FDR(confidence) <
    ``fdr_alpha`` computed WITHIN each type's spec>spec_min candidate set, then
    cap to ``top_n`` rows per type by rank.

    Args:
        df: DataFrame as returned by ``assemble_marker_table``.
        spec_min: minimum specificity score (exclusive); genes at or below this
            value are dropped. Default ``0.0`` keeps only genes with positive
            specificity.
        top_n: maximum number of rows to keep per cell type after all other
            filters. Pass ``None`` to keep all survivors. Default ``100``.
        fdr_alpha: if not ``None``, apply Benjamini-Hochberg FDR correction to
            the ``confidence`` column within each type's candidate set and keep
            only rows where the adjusted p-value is below this threshold.
        return_all: if ``True``, return ``df`` as-is with a reset index and skip
            all filtering. Useful for inspecting the full unfiltered table.

    Returns:
        A filtered ``pandas.DataFrame`` with the same columns as the input.
    """
    if return_all:
        return df.reset_index(drop=True)
    out = []
    for _, g in df.groupby("cell_type", sort=False):
        g = g[g["spec"] > spec_min].sort_values("rank")
        if fdr_alpha is not None and len(g):
            keep = _benjamini_hochberg(g["confidence"].to_numpy()) < fdr_alpha
            g = g[keep]
        if top_n is not None:
            g = g.head(top_n)
        out.append(g)
    if not out:  # only fires on empty input (no groups iterated)
        return df.iloc[:0].reset_index(drop=True)
    return pd.concat(out, ignore_index=True)


#: Minimum agreement with packaged symbols for identifying a symbol column.
_SYMBOL_AGREEMENT_THRESHOLD = 0.5


@functools.lru_cache(maxsize=12)
def _model_symbol_lookup(species: Species) -> Dict[str, str]:
    """Packaged Ensembl ID -> symbol mappings for genes with a real symbol.

    Genes the reference build leaves unnamed carry an empty symbol, or store the
    ID as their own symbol; both are dropped, because an array of Ensembl IDs
    would otherwise score matches against them and could be mistaken for a
    symbol array.

    Cache the read-only mapping.
    """
    df = _load_gene_mapping_table()
    prefix = _canonical_species(species)
    sym_col, ens_col = f"{prefix}_symbol", f"{prefix}_ensembl"
    return {str(e): str(s)
            for e, s in zip(df[ens_col].astype(str), df[sym_col].astype(str))
            if e and s and str(s) != str(e)}


def _detect_symbol_array_in_var(var, ensembl_ids, species):
    """Locate the array in ``var`` holding gene symbols, by agreement not by name.

    Score every column and the index on rows whose Ensembl IDs have known
    packaged symbols. Column naming is not used.

    Columns are scored before the index and ties are broken by first-seen, so a
    named column wins over an index holding the same symbols.

    Returns ``(values, source_name)``, or ``(None, None)`` when nothing clears
    :data:`_SYMBOL_AGREEMENT_THRESHOLD` — meaning ``var`` carries no symbols.
    """
    lookup = _model_symbol_lookup(species)
    rows, expected = [], []
    for i, gene_id in enumerate(ensembl_ids):
        symbol = lookup.get(gene_id)
        if symbol is not None:
            rows.append(i)
            expected.append(symbol)
    if not rows:
        return None, None
    rows = np.asarray(rows)
    expected = np.asarray(expected, dtype=str)

    candidates = [(str(c), var[c].values) for c in var.columns]
    candidates.append(("<index>", np.asarray(var.index)))
    best_values, best_name, best_score = None, None, 0.0
    for name, values in candidates:
        values = np.asarray(values)
        if values.shape[0] != len(ensembl_ids):
            continue
        score = float((values[rows].astype(str) == expected).mean())
        if score > best_score:
            best_values, best_name, best_score = values, name, score

    if best_score < _SYMBOL_AGREEMENT_THRESHOLD:
        return None, None
    return best_values.astype(str), best_name


def gene_symbols_for_ids(gene_ids, var, species=None):
    """Map model Ensembl gene IDs to display symbols using a ``var`` DataFrame.

    The symbols returned are the ones in ``var``, so they match the names in the
    caller's own data and can be searched for there. Which part of ``var`` holds
    them is decided by :func:`_detect_symbol_array_in_var` rather than by column
    name, so Cell Ranger output (symbols in ``gene_symbols``, Ensembl IDs in the
    index), CELLxGENE files (``feature_name``) and symbol-indexed objects all
    resolve without configuration.

    Only when ``var`` holds no symbols anywhere are HECTOR's packaged names used
    instead, logged as a warning. They are a last resort rather than the default
    because a reference build may name a gene differently from the caller's own
    annotation, and a symbol absent from their file is one they cannot look up.

    Ensembl version suffixes (e.g. ``ENSG00000123456.7``) are stripped on BOTH
    sides before matching, so versioned ``var`` IDs still resolve against
    unversioned model IDs (and vice versa). IDs with no symbol fall back to the
    ID string. Returns a list aligned with ``gene_ids``.
    """
    ids = _find_ensembl_id_array_in_var(var)
    if ids is None:
        return [str(g) for g in gene_ids]
    ids = [_strip_ensembl_version(str(e)) for e in np.asarray(ids)]
    if species is None:
        species = _species_from_ensembl_ids(ids)

    symbols, source = _detect_symbol_array_in_var(var, ids, species)
    if symbols is None:
        logger.warning(
            "No gene symbols found in adata.var; falling back to HECTOR's "
            "packaged gene names, which may differ from the names in your data.")
        lookup = _model_symbol_lookup(species)
        return [lookup.get(_strip_ensembl_version(str(g)), str(g))
                for g in gene_ids]

    logger.info("Gene symbols read from %s.",
                "the adata.var index" if source == "<index>"
                else f"adata.var[{source!r}]")
    ens2sym = {}
    for gene_id, symbol in zip(ids, symbols):
        if gene_id and gene_id.lower() != "nan":
            ens2sym[gene_id] = str(symbol)
    return [ens2sym.get(_strip_ensembl_version(str(g)), str(g)) for g in gene_ids]


# =============================================================================
# Gene co-expression networks + pathway enrichment (pure scipy/igraph/gProfiler)
# =============================================================================

class GeneNetworkBuilder:
    """Constructs gene interaction networks and detects modules.

    Builds gene-gene co-expression graphs from top attributed genes using
    Spearman correlation, applies Leiden community detection, and scores
    modules by attribution mass.
    """

    def __init__(self, top_k_genes=300, correlation_threshold=0.3,
                 p_value_threshold=0.05, leiden_resolution=1.0,
                 min_module_size=3):
        """
        Args:
            top_k_genes: Number of top genes to include in network.
            correlation_threshold: Absolute Spearman correlation cutoff.
            p_value_threshold: Significance threshold after Bonferroni correction.
            leiden_resolution: Resolution parameter for Leiden community detection.
            min_module_size: Minimum genes per module.
        """
        self.top_k_genes = top_k_genes
        self.correlation_threshold = correlation_threshold
        self.p_value_threshold = p_value_threshold
        self.leiden_resolution = leiden_resolution
        self.min_module_size = min_module_size

    def build_network(self, expression_matrix, top_gene_indices, attribution_scores):
        """Build gene network for one cell type.

        Args:
            expression_matrix: [num_cells, num_genes] raw expression for cells of this type.
            top_gene_indices: [top_k] indices of top attributed genes.
            attribution_scores: [num_genes] aggregated attribution scores.

        Returns:
            dict with 'edge_list' and 'modules', or None if too few genes.
        """
        k = min(self.top_k_genes, len(top_gene_indices))
        selected = top_gene_indices[:k]

        if len(selected) < self.min_module_size:
            logger.warning(f"Too few genes ({len(selected)}) for network construction")
            return None

        # Extract expression for selected genes (densify the small slice if sparse)
        expr_sub = expression_matrix[:, selected]  # [n_cells, k]
        if issparse(expr_sub):
            expr_sub = np.asarray(expr_sub.todense())

        # Drop genes with zero variance to avoid divide-by-zero in
        # np.corrcoef (called internally by spearmanr).
        gene_var = np.var(expr_sub, axis=0)
        nonzero_var = gene_var > 0
        if not np.all(nonzero_var):
            selected = selected[nonzero_var]
            expr_sub = expr_sub[:, nonzero_var]

        if len(selected) < self.min_module_size:
            logger.warning(
                f"Too few genes after variance filter ({len(selected)}) "
                f"for network construction"
            )
            return None

        n_genes = len(selected)
        n_pairs = n_genes * (n_genes - 1) // 2
        bonferroni_factor = max(n_pairs, 1)

        # Vectorized pairwise Spearman correlation over the [n_cells, k] matrix.
        result = scipy_stats.spearmanr(expr_sub)
        rho_matrix = getattr(result, 'statistic',
                     getattr(result, 'correlation', None))
        pval_matrix = result.pvalue

        if rho_matrix is None or np.ndim(rho_matrix) < 2:
            logger.warning("Spearmanr returned scalar (too few genes); skipping network")
            return {'edge_list': [], 'modules': []}

        # Bonferroni correction element-wise
        pval_bonf = np.minimum(pval_matrix * bonferroni_factor, 1.0)

        # Upper-triangle mask: passes both thresholds, NaN-safe
        mask = (
            np.abs(rho_matrix) >= self.correlation_threshold
        ) & (
            pval_bonf < self.p_value_threshold
        ) & ~np.isnan(rho_matrix)
        ii, jj = np.where(np.triu(mask, k=1))

        if len(ii) == 0:
            logger.warning("No significant edges found in gene network")
            return {'edge_list': [], 'modules': []}

        adj_dict = {i: [] for i in range(n_genes)}
        edge_list = []
        for i, j in zip(ii.tolist(), jj.tolist()):
            edge_list.append({
                'gene_a_idx': int(selected[i]),
                'gene_b_idx': int(selected[j]),
                'local_i': i,
                'local_j': j,
                'correlation': float(rho_matrix[i, j]),
                'p_value': float(pval_bonf[i, j]),
            })
            adj_dict[i].append(j)
            adj_dict[j].append(i)

        modules = self._detect_communities(adj_dict, n_genes, selected, attribution_scores)
        return {'edge_list': edge_list, 'modules': modules}

    def _detect_communities(self, adj_dict, n_genes, selected_indices, attribution_scores):
        """Run Leiden community detection, fall back to connected components."""
        try:
            import leidenalg
            import igraph as ig

            edges = []
            for node, neighbors in adj_dict.items():
                for neigh in neighbors:
                    if node < neigh:
                        edges.append((node, neigh))

            g = ig.Graph(n=n_genes, edges=edges, directed=False)
            partition = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self.leiden_resolution
            )
            membership = partition.membership
        except (ImportError, Exception) as e:
            logger.warning(f"Leiden failed ({e}), falling back to connected components")
            membership = self._connected_components(adj_dict, n_genes)

        module_map = {}
        for local_idx, mod_id in enumerate(membership):
            module_map.setdefault(mod_id, []).append(local_idx)

        modules = []
        mod_counter = 0
        for mod_id, local_indices in sorted(module_map.items()):
            if len(local_indices) < self.min_module_size:
                continue
            global_indices = [int(selected_indices[li]) for li in local_indices]
            attr_mass = float(np.sum([attribution_scores[gi] for gi in global_indices]))
            modules.append({
                'module_id': mod_counter,
                'gene_indices': global_indices,
                'attribution_mass': attr_mass,
            })
            mod_counter += 1
        return modules

    def _connected_components(self, adj_dict, n_genes):
        """Simple BFS connected components as Leiden fallback."""
        visited = [False] * n_genes
        membership = [-1] * n_genes
        comp_id = 0
        for start in range(n_genes):
            if visited[start]:
                continue
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                membership[node] = comp_id
                for neigh in adj_dict.get(node, []):
                    if not visited[neigh]:
                        visited[neigh] = True
                        queue.append(neigh)
            comp_id += 1
        return membership


class PathwayAnnotator:
    """Over-representation analysis on gene modules via Fisher's exact test
    with Benjamini-Hochberg FDR, against local GO/Reactome gene sets."""

    def __init__(self, go_gene_sets=None, reactome_gene_sets=None,
                 fdr_threshold=0.05):
        self.go_gene_sets = go_gene_sets
        self.reactome_gene_sets = reactome_gene_sets
        self.fdr_threshold = fdr_threshold
        if go_gene_sets is None and reactome_gene_sets is None:
            logger.warning("No pathway databases provided; enrichment will be skipped.")

    def annotate_module(self, module_genes, background_genes):
        """Enrichment for a single module. Returns a list of enriched-term dicts."""
        if self.go_gene_sets is None and self.reactome_gene_sets is None:
            return []
        module_set = set(module_genes)
        background_set = set(background_genes)
        n_background = len(background_set)
        n_module = len(module_set & background_set)
        if n_module == 0 or n_background == 0:
            return []

        raw_results = []
        if self.go_gene_sets is not None:
            for (term_id, term_name), term_genes in self.go_gene_sets.items():
                r = self._test_term(module_set, background_set, term_genes,
                                    term_id, term_name, 'GO', n_background, n_module)
                if r is not None:
                    raw_results.append(r)
        if self.reactome_gene_sets is not None:
            for (pw_id, pw_name), pw_genes in self.reactome_gene_sets.items():
                r = self._test_term(module_set, background_set, pw_genes,
                                    pw_id, pw_name, 'Reactome', n_background, n_module)
                if r is not None:
                    raw_results.append(r)
        if not raw_results:
            return []

        p_values = np.array([r['p_value_raw'] for r in raw_results])
        adjusted = self._benjamini_hochberg(p_values)
        enriched = []
        for r, adj_p in zip(raw_results, adjusted):
            if adj_p < self.fdr_threshold:
                enriched.append({
                    'term_id': r['term_id'], 'term_name': r['term_name'],
                    'source': r['source'], 'fold_enrichment': r['fold_enrichment'],
                    'p_value_adjusted': float(adj_p),
                    'overlapping_genes': r['overlapping_genes'],
                })
        enriched.sort(key=lambda x: x['p_value_adjusted'])
        return enriched

    def _test_term(self, module_set, background_set, term_genes,
                   term_id, term_name, source, n_background, n_module):
        """Fisher's exact test for one term."""
        term_in_bg = term_genes & background_set
        if len(term_in_bg) == 0:
            return None
        overlap = module_set & term_in_bg
        a = len(overlap)
        if a == 0:
            return None
        b = max(n_module - a, 0)
        c = max(len(term_in_bg) - a, 0)
        d = max(n_background - n_module - c, 0)
        _, p_value = scipy_stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        expected = (n_module * len(term_in_bg)) / max(n_background, 1)
        fold_enrichment = a / max(expected, 1e-10)
        return {
            'term_id': term_id, 'term_name': term_name, 'source': source,
            'fold_enrichment': float(fold_enrichment), 'p_value_raw': float(p_value),
            'overlapping_genes': sorted(list(overlap)),
        }

    @staticmethod
    def _benjamini_hochberg(p_values):
        """Benjamini-Hochberg FDR correction. Delegates to the module-level implementation."""
        return _benjamini_hochberg(p_values)


class GProfilerAnnotator:
    """Online over-representation analysis via g:Profiler's g:GOSt endpoint.

    Accepts Ensembl gene IDs directly. Falls back gracefully (empty result) if
    ``gprofiler-official`` is not installed or the API is unreachable.
    """

    def __init__(self, organism='hsapiens', sources=None, fdr_threshold=0.05):
        self.organism = organism
        self.sources = sources or ['GO:BP', 'REAC']
        self.fdr_threshold = fdr_threshold
        self._gp = None
        self._available = None

    def _ensure_client(self):
        if self._available is not None:
            return self._available
        try:
            from gprofiler import GProfiler
            self._gp = GProfiler(return_dataframe=True)
            self._available = True
        except ImportError:
            logger.warning("gprofiler-official not installed; skipping online enrichment.")
            self._available = False
        return self._available

    def annotate_module(self, module_gene_ids, background_gene_ids):
        if not self._ensure_client() or len(module_gene_ids) == 0:
            return []
        try:
            result_df = self._gp.profile(
                organism=self.organism, query=list(module_gene_ids),
                background=list(background_gene_ids), domain_scope='custom',
                sources=self.sources, user_threshold=self.fdr_threshold,
                significance_threshold_method='fdr', no_evidences=False,
            )
        except Exception as e:
            logger.warning(f"g:Profiler API call failed: {e}")
            return []
        if result_df is None or result_df.empty:
            return []
        enriched = []
        for _, row in result_df.iterrows():
            query_size = row.get('query_size', len(module_gene_ids))
            term_size = row.get('term_size', 1)
            intersection_size = row.get('intersection_size', 0)
            effective_domain = row.get('effective_domain_size', len(background_gene_ids))
            expected = (query_size * term_size) / max(effective_domain, 1)
            fold_enrichment = intersection_size / max(expected, 1e-10)
            intersections = row.get('intersections', [])
            if isinstance(intersections, str):
                overlapping = [g.strip() for g in intersections.split(',') if g.strip()]
            elif isinstance(intersections, (list, tuple)):
                overlapping = list(intersections)
            else:
                overlapping = []
            enriched.append({
                'term_id': row.get('native', row.get('term_id', '')),
                'term_name': row.get('name', row.get('term_name', '')),
                'source': row.get('source', ''),
                'fold_enrichment': float(fold_enrichment),
                'p_value_adjusted': float(row.get('p_value', 1.0)),
                'overlapping_genes': sorted(overlapping),
            })
        enriched.sort(key=lambda x: x['p_value_adjusted'])
        return enriched


def load_gene_sets_from_gmt(gmt_path):
    """Load pathway gene sets from a local GMT file into a
    ``{(term_id, term_name): set(genes)}`` dict (or None on failure)."""
    if not os.path.isfile(gmt_path):
        logger.error(f"GMT file not found: {gmt_path}")
        return None
    gene_sets = {}
    n_terms = 0
    with open(gmt_path, 'r') as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            term_name = parts[0]
            description = parts[1] if parts[1] else term_name
            genes = set(g.strip() for g in parts[2:] if g.strip())
            if not genes:
                continue
            term_id = (description if ':' in description or description.startswith('R-')
                       else term_name)
            gene_sets[(term_id, term_name)] = genes
            n_terms += 1
    if n_terms == 0:
        logger.warning(f"No valid terms parsed from GMT file: {gmt_path}")
        return None
    logger.info(f"Loaded {n_terms} gene sets from {gmt_path}")
    return gene_sets


# ============================================================================
# Preflight peak-memory estimation
# The predictor gathers live inputs; these helpers project memory from supplied values.
# ============================================================================

_GB = 1024 ** 3
_MB = 1024 ** 2
_INT32_MAX = 2 ** 31 - 1


@dataclass
class PreflightReport:
    """Result of a peak-memory projection. Byte fields are ints (bytes)."""
    backend: str
    resident_bytes: int
    available_bytes: int
    peak_increment_bytes: int
    projected_peak_bytes: int
    ledger: List[Tuple[str, int, bool]]         # (stage, increment_bytes, will_run)
    unused_copies: List[Tuple[str, int, str]]   # (slot, bytes, reason)
    verdict: str                                 # 'ok' | 'caution' | 'likely_oom'
    disk_note: Optional[str]
    message: str
    grit_message: str = ""


def _sparse_bytes(nnz: int, n_rows: int, idx_bytes: int) -> int:
    """Bytes for a float32 CSR matrix: data + indices + indptr."""
    return nnz * 4 + nnz * idx_bytes + (n_rows + 1) * idx_bytes


def estimate_prediction_memory(
    *,
    n_cells, nnz_reordered, latent_dim, n_active_classes,
    encoder_fixed_floor, encoder_per_cell, batch_size, k_neighbors,
    use_grit, embeddings_cached, backend,
    available_bytes, resident_bytes,
    unused_copy_sizes=None, safety_fraction=0.85, projection_margin=1.3,
    fixed_host_overhead=0,        # CUDA: anchor .numpy() backup + TF/CUDA host staging + arena
    grit_score_multiplier=0,      # CUDA: host copies of the score matrix GRIT holds (filtered+refined+propagate)
    encoder_increment_bytes=None, # Already-guarded additional host/unified memory
    grit_increment_bytes=None,    # Already-guarded tiled-GRIT host/unified memory
) -> PreflightReport:
    """Project the additional host RAM needed by prediction stages.

    CUDA encoder and graph working sets reside in VRAM and are sized separately.
    MLX and CPU working sets use host or unified RAM and count here. On CUDA,
    fixed_host_overhead covers anchor backups and staging; grit_score_multiplier
    covers host score copies.

    Decision rule:
        peak_increment = max(stage_increment)
        peak_increment > available_bytes -> 'likely_oom'
        unguarded stage > safety_fraction * available_bytes -> 'caution'
        otherwise -> 'ok'

    resident_bytes is already allocated; compare only new stage allocations
    against currently available memory. projection_margin inflates analytical
    estimates. encoder_increment_bytes and grit_increment_bytes already include
    allocator guards and are not inflated again.
    """
    unused_copy_sizes = list(unused_copy_sizes or [])
    idx = 4 if nnz_reordered < _INT32_MAX else 8
    x_norm = _sparse_bytes(nnz_reordered, n_cells, idx)
    emb = n_cells * latent_dim * 4
    encoder_ws = max(int(encoder_fixed_floor), int(encoder_per_cell) * int(batch_size))
    grit_graph = n_cells * k_neighbors * (4 + 8) if use_grit else 0
    score_bytes = n_cells * n_active_classes * 4

    memmap_threshold = max(min(int(available_bytes * 0.25), _GB), 256 * _MB)
    # Count the full score matrix when GRIT is configured to hold host copies.
    # Otherwise, exclude large memmapped buffers from this allocation estimate;
    # their resident pages and OS file cache can still consume physical memory.
    grit_holds_score = use_grit and grit_score_multiplier > 0
    if grit_holds_score:
        score_in_ram = score_bytes
        disk_note = None
    else:
        score_in_ram = 0 if score_bytes > memmap_threshold else score_bytes
        disk_note = (None if score_in_ram else
                     f"score matrix ~{score_bytes / _GB:.1f} GB spills to disk (memmap), not RAM")

    run_encoder = not embeddings_cached
    run_preprocess = run_encoder or use_grit
    x_norm_new = x_norm if run_preprocess else 0
    emb_new = emb if run_encoder else 0
    oh = int(fixed_host_overhead)

    # The encoder GPU working set and the GRIT kNN graph are VRAM allocations on
    # CUDA (HECTOR sizes/retries them against VRAM itself, and a VRAM OOM is
    # catchable), so the guard does not count them against host RAM there. On
    # MLX (unified) / CPU they live in host RAM, so they DO count.
    on_device = backend == "cuda"
    host_ws = 0 if on_device else encoder_ws
    host_knn = 0 if on_device else grit_graph

    raw_ledger = [
        ("preprocess (reorder+normalize)", (x_norm + x_norm) if run_preprocess else 0, run_preprocess),
        (
            "encoder",
            (int(encoder_increment_bytes) if encoder_increment_bytes is not None
             else x_norm_new + emb_new + host_ws + oh) if run_encoder else 0,
            run_encoder,
        ),
        ("classify", x_norm_new + emb_new + score_in_ram + oh, True),
        (
            "grit",
            (x_norm_new + emb_new
             + (int(grit_increment_bytes) if grit_increment_bytes is not None
                else host_knn + grit_score_multiplier * score_bytes + oh))
            if use_grit else 0,
            use_grit,
        ),
    ]
    guarded_stages = {
        "encoder" if encoder_increment_bytes is not None else "",
        "grit" if grit_increment_bytes is not None else "",
    }
    ledger = [
        (
            stage,
            int(increment) if stage in guarded_stages
            else int(round(increment * projection_margin)),
            will_run,
        )
        for stage, increment, will_run in raw_ledger
    ]

    peak_increment = max(inc for _s, inc, _r in ledger)
    projected_peak = resident_bytes + peak_increment

    if any(increment > available_bytes for _stage, increment, runs in ledger if runs):
        verdict = "likely_oom"
    elif any(
        increment > safety_fraction * available_bytes
        for stage, increment, runs in ledger
        if runs and stage not in guarded_stages
    ):
        verdict = "caution"
    else:
        verdict = "ok"

    message = _format_message(
        backend=backend, verdict=verdict, peak_increment=peak_increment,
        projected_peak=projected_peak, resident=resident_bytes,
        available=available_bytes, ledger=ledger, unused_copies=unused_copy_sizes,
        use_grit=use_grit, x_norm=x_norm, disk_note=disk_note,
        projection_margin=projection_margin,
    )
    return PreflightReport(
        backend=backend, resident_bytes=resident_bytes, available_bytes=available_bytes,
        peak_increment_bytes=peak_increment, projected_peak_bytes=projected_peak,
        ledger=ledger, unused_copies=unused_copy_sizes, verdict=verdict,
        disk_note=disk_note, message=message,
    )


def _gb(n: int) -> str:
    """Format a byte count as a human-readable gigabytes string (e.g. '4.2 GB')."""
    return f"{n / _GB:.1f} GB"


def _format_message(
    *, backend, verdict, peak_increment, projected_peak, resident, available,
    ledger, unused_copies, use_grit, x_norm, disk_note, projection_margin=1.0,
) -> str:
    """Return a one-line host-memory warning, or an empty string when the run fits.

    Note loaded but unused expression copies without advising deletion. Suggest
    use_grit=False only when GRIT dominates the projected memory shortfall.
    """
    if verdict == "ok":
        return ""

    if verdict == "likely_oom":
        head = (f"⚠️ predict() may run out of memory: it needs "
                f"≈ {_gb(peak_increment)} more, but only ≈ {_gb(available)} is free.")
    else:  # caution
        head = (f"⚠️ memory may be tight: predict() needs "
                f"≈ {_gb(peak_increment)} more, with ≈ {_gb(available)} free.")

    # Advise disabling GRIT ONLY when GRIT is the dominant stage driving the OOM
    # (on CUDA it holds several score-matrix copies in RAM). Do NOT advise it when
    # some other stage dominates (e.g. the encoder on MLX).
    if verdict == "likely_oom" and use_grit:
        if ledger and max(ledger, key=lambda t: t[1])[0] == "grit":
            head += " Disabling GRIT (use_grit=False) would cut the biggest cost."

    if unused_copies:
        slot, nbytes, _reason = max(unused_copies, key=lambda t: t[1])
        head += (f" (Note: {slot}, {_gb(nbytes)}, is loaded but not read by "
                 f"predict().)")
    return head
