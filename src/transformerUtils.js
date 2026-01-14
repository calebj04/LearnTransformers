export const getTokenColor = (index) => {
  const colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981"];
  return colors[index % colors.length];
};

export const getEmbeddingHeight = (token, dim) => {
  const seed = token.charCodeAt(0) * (dim + 1);
  return 30 + (seed % 50);
};

export const getAttentionWeight = (from, to) => {
  if (from === to) return 0.4;
  if (Math.abs(from - to) === 1) return 0.3;
  return 0.1 + Math.random() * 0.2;
};
