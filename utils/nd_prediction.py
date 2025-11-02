"""
N-Dimensional Autoregressive Prediction

Instead of linearizing N-D data into 1D sequences, this maintains the true
N-dimensional structure of the prediction task.

Core Idea:
    In 1D: Given [t-k, ..., t-1], predict t
    In 2D: Given filled region R ⊂ Z^2, predict frontier ∂R
    In N-D: Given filled region R ⊂ Z^N, predict frontier ∂R

This is a more natural formulation for spatial data where:
- Context is a spatial neighborhood, not a sequence
- Prediction target is the boundary/frontier, not "next token"
- Multiple predictions can happen in parallel (frontier pixels)
