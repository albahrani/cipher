/**
 * TF-IDF Embedder Implementation
 *
 * Provides embedding functionality using TF-IDF vectorization.
 * This is a cost-free, non-LLM embedding backend.
 */
import { Embedder, EmbeddingConfig } from './types.js';
import { logger } from '../../../logger/index.js';

// Simple TF-IDF implementation for demonstration
class TfIdf {
  private documents: string[] = [];
  public vocabulary: Set<string> = new Set();
  private idf: Map<string, number> = new Map();

  addDocument(doc: string) {
    this.documents.push(doc);
    doc.split(/\W+/).forEach(word => {
      if (word) this.vocabulary.add(word.toLowerCase());
    });
  }

  finalize() {
    const docCount = this.documents.length;
    this.vocabulary.forEach(word => {
      let count = 0;
      this.documents.forEach(doc => {
        if (doc.toLowerCase().includes(word)) count++;
      });
      this.idf.set(word, Math.log(docCount / (1 + count)));
    });
  }

  vectorize(doc: string): number[] {
    const tf: Map<string, number> = new Map();
    const words = doc.split(/\W+/).map(w => w.toLowerCase());
    this.vocabulary.forEach(word => {
      tf.set(word, words.filter(w => w === word).length / words.length);
    });
    return Array.from(this.vocabulary).map(word => (tf.get(word) || 0) * (this.idf.get(word) || 0));
  }
}

export class TfIdfEmbedder implements Embedder {
  private readonly config: EmbeddingConfig;
  private readonly tfidf: TfIdf;

  constructor(config: EmbeddingConfig) {
    this.config = config;
    this.tfidf = new TfIdf();
    logger.debug('[TFIDF] Embedder initialized');
  }

  // For demonstration, add documents before embedding
  addDocuments(docs: string[]) {
    docs.forEach(doc => this.tfidf.addDocument(doc));
    this.tfidf.finalize();
  }

  async embed(text: string): Promise<number[]> {
    return this.tfidf.vectorize(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return texts.map(text => this.tfidf.vectorize(text));
  }

  getDimension(): number {
    return this.tfidf ? Array.from(this.tfidf.vocabulary).length : 0;
  }

  getConfig(): EmbeddingConfig {
    return this.config;
  }

  async isHealthy(): Promise<boolean> {
    return true;
  }

  async disconnect(): Promise<void> {
    // No resources to clean up
  }
}
