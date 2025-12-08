import React from "react";

// Utility to generate a random embedding vector
const generateRandomEmbedding = (dim = 8) => {
  return Array.from({ length: dim }, () => Math.random());
};

// Positional encoding (simplified version for demonstration)
const addPositionalEncoding = (embedding, position) => {
  return embedding.map((value, i) => value + Math.sin(position / (i + 1)));
};

const EmbeddingPositionalModule = ({ tokens }) => {
  const embeddingDim = 8;

  if (!tokens || tokens.length === 0) {
    return (
      <p style={{ padding: "20px" }}>
        No tokens available. Please tokenize some text first.
      </p>
    );
  }

  // Generate embeddings with positional encodings
  const tokenEmbeddings = tokens.map((token, idx) => {
    const emb = generateRandomEmbedding(embeddingDim);
    const embWithPos = addPositionalEncoding(emb, idx);
    return { ...token, embedding: embWithPos };
  });

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2>4.2.2 Embedding and Positional Encoding Module</h2>
      <p>
        Each token ID is mapped to a continuous vector representation.
        Positional encodings are added to token embeddings to preserve sequence
        order. This module visualizes the embeddings as colored bars and shows
        how positional information is combined.
      </p>

      <div style={{ marginTop: "20px" }}>
        {tokenEmbeddings.map((token, idx) => (
          <div key={idx} style={{ marginBottom: "20px" }}>
            <p>
              {" "}
              {token.word} (ID: {token.id})
            </p>
            <div style={{ display: "flex" }}>
              {token.embedding.map((val, i) => (
                <div
                  key={i}
                  style={{
                    width: "20px",
                    height: "50px",
                    marginRight: "2px",
                    backgroundColor: `rgb(${Math.floor(
                      val * 255
                    )}, ${Math.floor((1 - val) * 255)}, 150)`,
                    transition: "all 0.3s",
                  }}
                  title={`Value: ${val.toFixed(3)}`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <p style={{ fontStyle: "italic" }}>
        Note: Embeddings and positional encodings are simulated for
        visualization purposes.
      </p>
    </div>
  );
};

export default EmbeddingPositionalModule;
