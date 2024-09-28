# Jobs to enrich dataset

To further enrich the dataset and improve the quality of recommendations, we can add several additional features. These new features will help capture more nuanced aspects of user preferences and track characteristics, leading to a more personalized and accurate recommendation system.

## Addition Features

### Artist Popularity Score

**Feature Description**: Add an artist popularity score based on the average popularity of all tracks by the same artist. This allows for prioritizing tracks by more well-known or popular artists.

**Benefit**: If a user likes a certain artist, it helps to suggest more tracks from the same or similarly popular artists, adding a layer of personalization based on artist preferences.

### Genre Popularity Score

**Feature Description**: Calculate a popularity score for each genre based on the average popularity of tracks within that genre. This will allow the system to prioritize recommendations from popular genres if the user does not specify a genre preference.

**Benefit**: Helps to recommend tracks from genres that are generally popular, potentially improving user satisfaction by aligning recommendations with broader trends.

### Embedding and HNSW index

**Feature Description**: The purpose of generating embeddings is to transform raw data into a compact, numerical representation that captures the essential patterns and relationships within the data. These embeddings can be used for various machine learning tasks and analyses.

**Similarity Search**: Embeddings are helpful when measuring the similarity between different data points. For example, in a music recommendation system, embeddings help identify songs that are similar in terms of features like tempo, energy, or loudness.

**HNSW Index**: HNSW (Hierarchical Navigable Small World) is a highly efficient algorithm used for approximate nearest neighbor (ANN) search in high-dimensional spaces. The goal of nearest neighbor search is to quickly find data points (or vectors) that are most similar to a given query point based on some distance metric (e.g., Euclidean distance, cosine similarity). Here we choose cosine similarity as it space.
