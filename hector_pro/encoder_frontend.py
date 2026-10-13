"""TensorFlow gene-token encoder layers and checkpoint-loading helpers.

The front-end uses frozen protein embeddings, species-specific gene mappings,
FiLM tokenization, and learned attention pooling. It is independent of training
modules and loads the checkpoint's shared-encoder weights positionally.
"""

import numpy as np
import tensorflow as tf


def sparsify_topk(x: tf.Tensor, k: int):
    """Pick each cell's top-k genes by expression.

    Args:
        x: dense expression [B, num_genes], non-negative (log1p-normalized).
        k: number of genes to keep per cell, at most num_genes.
    Returns:
        gene_idx [B, k] int32 column indices, values [B, k] in x's dtype,
        mask [B, k] float32 (1 for detected genes, 0 for zero-padding).

    Zero-expression entries are retained when fewer than k genes are detected,
    but excluded by the returned mask.
    """
    values, indices = tf.math.top_k(x, k=k, sorted=False)
    mask = tf.cast(values > 0.0, tf.float32)
    return tf.cast(indices, tf.int32), values, mask


class GeneTokenFiLM(tf.keras.layers.Layer):
    """Turn (global_idx, expression) into FiLM tokens from a frozen embedding table.

    token = LayerNorm( gamma(E[global_idx]) * expression + beta(E[global_idx]) )
    gamma/beta are computed for the shared table in each call, then gathered
    per token without gathering a per-cell copy of the raw embedding table.

    The table is a single flat [R_prot+1, emb_dim] array. The caller
    (GeneTokensFrontEnd) maps (species, column) -> a global row via the M map, so this
    layer is species- and column-agnostic: it only gathers rows by global_idx.
    """

    def __init__(self, embedding_table: np.ndarray, token_dim: int,
                 eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        arr = np.asarray(embedding_table, dtype="float32")
        if arr.ndim != 2:
            raise ValueError(
                "embedding_table must be a flat [R_prot+1, emb_dim] 2-D array; "
                f"got shape {arr.shape}")
        self.num_table_rows, self.emb_dim = arr.shape
        self._init_table = arr
        self.token_dim = int(token_dim)
        self.eps = eps

    def build(self, input_shape):
        # Initialize the frozen table with its supplied values when the weight
        # is created, so graph tracing sees the checkpoint embeddings.
        self.E = self.add_weight(
            name="protein_emb",
            shape=(self.num_table_rows, self.emb_dim),
            initializer=tf.constant_initializer(self._init_table),
            trainable=False,
        )
        self.film = tf.keras.layers.Dense(2 * self.token_dim, name="film_proj")
        self.ln = tf.keras.layers.LayerNormalization(epsilon=self.eps, name="token_ln")
        super().build(input_shape)

    def call(self, inputs):
        # inputs = (global_idx, value): global_idx [B, K] int into the flat table's
        # [R_prot+1] rows; value [B, K] float.
        global_idx, value = inputs
        gb = self.film(self.E)                              # [num_table_rows, 2*token_dim]
        gamma_all, beta_all = tf.split(gb, 2, axis=-1)      # [num_table_rows, token_dim]
        gamma = tf.gather(gamma_all, global_idx)            # [B, K, token_dim]
        beta = tf.gather(beta_all, global_idx)              # [B, K, token_dim]
        token = gamma * value[..., None] + beta             # [B, K, token_dim]
        return self.ln(token)


class WithinCellEncoder(tf.keras.layers.Layer):
    """Attention-pool a cell's gene tokens -> cell vector.

    A single learned query attends over the cell's masked gene tokens, and
    the pooled result is projected to the cell vector.
    """

    def __init__(self, token_dim, num_heads=8, output_dim=768,
                 dropout=0.1, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.dropout = dropout
        self.eps = eps

    def build(self, input_shape):
        self.pool_query = self.add_weight(
            name="pool_query", shape=(1, 1, self.token_dim), initializer="glorot_uniform",
        )
        self.pool_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.token_dim // self.num_heads,
            dropout=self.dropout,
        )
        self.proj = tf.keras.layers.Dense(self.output_dim, name="cell_proj")
        super().build(input_shape)

    def call(self, tokens, mask, training=None):
        # mask [B, K] (1 real, 0 pad) -> boolean key mask for the pooling attention.
        key_mask = tf.cast(mask, tf.bool)                          # [B, K]
        q = tf.tile(self.pool_query, [tf.shape(tokens)[0], 1, 1])  # [B, 1, token_dim]
        pooled = self.pool_attn(q, tokens, attention_mask=key_mask[:, None, :],
                                training=training)                 # [B, 1, token_dim]
        pooled = tf.squeeze(pooled, axis=1)                        # [B, token_dim]
        return self.proj(pooled)                                   # [B, output_dim]


class GeneTokensFrontEnd(tf.keras.layers.Layer):
    """Expression plus species index [B, num_genes + 1] -> cell vector [B, output_dim].

    Select each cell's top-k genes, build FiLM tokens from frozen protein
    embeddings, and aggregate them with learned attention pooling. output_dim
    must match the encoder's shared_dim.
    """

    def __init__(self, embedding_table, col_to_row, k, token_dim, output_dim,
                 num_heads=8, dropout=0.1, gene_token_chunk=512, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)
        # Bound intermediate [chunk, K, token_dim] token tensors. The default
        # leaves room below the int32 element-count limit for K=2048 and
        # token_dim=1024. A non-positive value disables chunking.
        self.gene_token_chunk = int(gene_token_chunk)
        # M[s, c] maps species s and model column c to an embedding-table row.
        # Validate row bounds at construction.
        M = np.asarray(col_to_row)
        if M.ndim != 2 or not np.issubdtype(M.dtype, np.integer):
            raise ValueError(
                "col_to_row (M) must be a 2-D integer array [num_species, R_cols]; "
                f"got shape {M.shape}, dtype {M.dtype}")
        table_rows = np.asarray(embedding_table).shape[0]
        if M.min() < 0 or M.max() >= table_rows:
            raise ValueError(
                f"col_to_row row index out of range: [{int(M.min())}, {int(M.max())}] "
                f"must lie in [0, {table_rows}) (R_prot rows of the embedding table)")
        self._M_init = M.astype(np.int32)
        self.tokenizer = GeneTokenFiLM(embedding_table, token_dim=token_dim,
                                       name="gene_token_film")
        self.cell_encoder = WithinCellEncoder(
            token_dim=token_dim, num_heads=num_heads,
            output_dim=output_dim, dropout=dropout, name="within_cell_encoder",
        )

    def build(self, input_shape):
        # Store the species map as a non-trainable checkpoint weight. An explicit
        # int32 dtype preserves row indices under mixed precision.
        self.M = self.add_weight(
            name="col_to_row", shape=self._M_init.shape,
            initializer=tf.constant_initializer(self._M_init),
            trainable=False, dtype="int32",
        )
        super().build(input_shape)

    def call(self, inputs, training=None, chunk_size=None):
        # inputs [B, num_genes + 1] contains expression followed by the per-cell
        # species index, matching the encoder's single-tensor input contract.
        x = inputs[:, :-1]
        species_id = tf.cast(inputs[:, -1], tf.int32)                  # [B]
        num_genes = tf.shape(x)[1]
        k = tf.minimum(self.k, num_genes)
        gene_idx, value, mask = sparsify_topk(x, k=k)
        # Gather flat-table rows through the species-specific column map.
        sp = tf.broadcast_to(species_id[:, None], tf.shape(gene_idx))         # [B, K]
        global_idx = tf.gather_nd(self.M, tf.stack([sp, gene_idx], axis=-1))  # [B, K] = M[species, col]

        cs = self.gene_token_chunk if chunk_size is None else int(chunk_size)
        if cs and cs > 0:
            # Tokenize and pool at most cs cells at a time. tf.while_loop supports
            # dynamic batches in graph mode; infer_shape=False permits a smaller
            # final chunk.
            B = tf.shape(global_idx)[0]
            n_chunks = (B + cs - 1) // cs
            out = tf.TensorArray(
                self.compute_dtype, size=n_chunks, infer_shape=False,
                element_shape=tf.TensorShape([None, self.cell_encoder.output_dim]))

            def _cond(i, _out):
                return i < n_chunks

            def _body(i, _out):
                s = i * cs
                e = tf.minimum(s + cs, B)
                tok = self.tokenizer((global_idx[s:e], value[s:e]))
                res = self.cell_encoder(tok, mask[s:e], training=training)
                return i + 1, _out.write(i, res)

            _, out = tf.while_loop(_cond, _body, [tf.constant(0), out])
            return out.concat()

        tokens = self.tokenizer((global_idx, value))
        return self.cell_encoder(tokens, mask, training=training)


def build_frontend(protein_emb, col_to_row, k=2048, token_dim=1024, output_dim=1024, num_heads=8):
    """Construct and build the front-end with the checkpoint's frozen table and map."""
    front = GeneTokensFrontEnd(
        embedding_table=protein_emb, col_to_row=col_to_row,
        k=k, token_dim=token_dim, output_dim=output_dim, num_heads=num_heads, dropout=0.0,
    )
    n_genes = int(col_to_row.shape[1])
    _ = front(tf.zeros([2, n_genes + 1], tf.float32), training=False)  # build sublayers
    return front


def load_frontend_weights(front, shared_weights):
    """Load shared_encoder weights 0 through 16 in checkpoint order."""
    front.set_weights([np.asarray(w) for w in shared_weights[:17]])
