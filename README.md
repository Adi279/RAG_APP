


## How hybrid search works here

Pinecone stores one dotproduct index holding both a dense vector and a
sparse (BM25) vector per chunk. At query time we scale the dense query vector
by `alpha` and the sparse query vector by `(1 - alpha)` before sending both to
`index.query()` — this is Pinecone's documented convex-combination approach to
hybrid search. The two sliders in the sidebar are linked so moving one
auto-sets the other to `1 - value`.


