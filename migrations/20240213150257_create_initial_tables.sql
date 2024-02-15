-- Make sure pgvectgor is installed
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop table to start fresh every time (for development purposes, may be removed later)
DROP TABLE IF EXISTS rippy;

-- Create tabel for embeddings
-- The vector size is dependend on the embedding model used!
CREATE TABLE IF NOT EXISTS rippy (id bigserial PRIMARY KEY, chunk text, embedding vector(1024));

