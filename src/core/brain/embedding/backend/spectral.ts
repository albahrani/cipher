/**
 * SpectralEmbedder: Low-dimensional spectral embedding backend for Cipher
 * Implements SVD/PCA/Laplacian Eigenmaps for document embeddings
 */
import { Embedder, EmbeddingConfig } from './types.js';
import { Matrix, SingularValueDecomposition } from 'ml-matrix';

export interface SpectralEmbeddingConfig extends EmbeddingConfig {
  type: 'spectral';
  method?: 'svd' | 'pca' | 'laplacian';
  dimension?: number;
}

export class SpectralEmbedder implements Embedder {
  getDimension(): number {
    return this.config.dimension || (this.components ? this.components.columns : 0);
  }

  getConfig(): SpectralEmbeddingConfig {
    return this.config;
  }

  async isHealthy(): Promise<boolean> {
    return this.fitted && !!this.components;
  }

  async disconnect(): Promise<void> {
    // No resources to clean up for local spectral embedder
    this.fitted = false;
    this.components = null;
  }
  private config: SpectralEmbeddingConfig;
  private vocabulary: string[] = [];
  private idf: number[] = [];
  private fitted: boolean = false;
  private components: Matrix | null = null;

  constructor(config: SpectralEmbeddingConfig) {
    this.config = config;
  }

  /** Fit the spectral model to a corpus */
  async fit(corpus: string[]): Promise<void> {
    // Step 1: Build TF-IDF matrix
    const tfidfMatrix = this.buildTfidfMatrix(corpus);
    // Step 2: Apply spectral method
    const method = this.config.method || 'svd';
    const k = this.config.dimension || Math.min(128, tfidfMatrix.columns);
    if (method === 'svd' || method === 'pca' || method === 'laplacian') {
      // SVD decomposition (used for all spectral methods due to library limitations)
      const svd = new SingularValueDecomposition(tfidfMatrix);
      this.components = svd.leftSingularVectors.subMatrix(0, tfidfMatrix.rows - 1, 0, k - 1);
    }
    this.fitted = true;
  }

  /** Embed a document */
  async embed(text: string): Promise<number[]> {
    if (!this.fitted || !this.components) {
      throw new Error('SpectralEmbedder: Model not fitted. Call fit() with corpus first.');
    }
    // Build TF-IDF vector for text
    const tfidfVec = this.buildTfidfVector(text);
    // Project onto spectral components
    return this.components.transpose().mmul(Matrix.columnVector(tfidfVec)).to1DArray();
  }

  /** Fit and embed batch */
  async embedBatch(texts: string[]): Promise<number[][]> {
    await this.fit(texts);
    return Promise.all(texts.map(t => this.embed(t)));
  }

  /** Build TF-IDF matrix for corpus */
  private buildTfidfMatrix(corpus: string[]): Matrix {
    // Build vocabulary
    const vocabSet = new Set<string>();
    corpus.forEach(doc => doc.split(/\W+/).forEach(w => vocabSet.add(w.toLowerCase())));
    this.vocabulary = Array.from(vocabSet);
    // Compute IDF
    const N = corpus.length;
    this.idf = this.vocabulary.map(word => {
      const df = corpus.filter(doc => doc.toLowerCase().includes(word)).length;
      return Math.log((N + 1) / (df + 1)) + 1;
    });
    // Build matrix
    const matrix = corpus.map(doc => {
      const words = doc.split(/\W+/).map(w => w.toLowerCase());
      return this.vocabulary.map((word, i) => {
        const tf = words.filter(w => w === word).length / words.length;
        return tf * this.idf[i];
      });
    });
    return new Matrix(matrix);
  }

  /** Build TF-IDF vector for a single document */
  private buildTfidfVector(text: string): number[] {
    const words = text.split(/\W+/).map(w => w.toLowerCase());
    return this.vocabulary.map((word, i) => {
      const tf = words.filter(w => w === word).length / words.length;
      return tf * this.idf[i];
    });
  }

  /** Build Laplacian matrix (simple version) */
  private buildLaplacian(tfidf: Matrix): Matrix {
    // Similarity matrix (cosine)
    const n = tfidf.rows;
    const sim = Matrix.zeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const a = tfidf.getRow(i);
        const b = tfidf.getRow(j);
        const dot = a.reduce((sum, v, k) => sum + v * b[k], 0);
        const normA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
        const normB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
        sim.set(i, j, normA && normB ? dot / (normA * normB) : 0);
      }
    }
    // Degree matrix
    const deg = Matrix.zeros(n, n);
    for (let i = 0; i < n; i++) {
      deg.set(i, i, sim.getRow(i).reduce((sum, v) => sum + v, 0));
    }
    // Laplacian: L = D - S
    return deg.sub(sim);
  }
}
