# Self-Supervised Learning Task Brainstorm

Comprehensive collection of pretext tasks for learning rich visual representations from raw image and video data.

## The Core Question: Image Equivalents to "Next Token Prediction"

**Key Insight**: Next token prediction is the foundation of language model success. What's the vision equivalent?

### Why Next Token Prediction Works for Language

1. **Natural sequential structure**: Text has inherent left-to-right ordering
2. **Self-supervised**: Every token provides supervision for the next
3. **Scales perfectly**: More data = more training signal
4. **Forces semantic understanding**: Can't predict next word without understanding context
5. **Autoregressive**: Builds on previous predictions

### The Image Challenge: Constructing Sequential Tokens

**Problem**: Images are inherently 2D/3D spatial structures, not sequential. How do we impose or discover a meaningful ordering?

### Approaches to Sequential Token Construction

#### 1. **Raster/Scanline Order** (Naive but functional)

**What is Raster Ordering?**

Raster ordering (also called scanline order) is how old CRT monitors drew images: left-to-right, top-to-bottom, like reading English text.

```
Visual Example:

Image (4x4):        Raster Order:

A B C D            1→ 2→ 3→ 4
E F G H            5→ 6→ 7→ 8
I J K L            9→10→11→12
M N O P            13→14→15→16

Prediction sequence: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P
```

**How it works for next token prediction**:
```python
# Flatten image to 1D sequence in raster order
for row in range(height):
    for col in range(width):
        pixel = image[row, col]
        # Predict this pixel from all previous pixels
        prediction = model(pixels_so_far)
        pixels_so_far.append(pixel)

# Position 0: Predict pixel A (no context)
# Position 1: Predict B given A
# Position 2: Predict C given A, B
# Position 5: Predict F given A, B, C, D, E
# etc.
```

**Why "Raster"?**

Named after raster graphics/displays:
- CRT electron beam scans left→right (horizontal sweep)
- Moves down one line (vertical refresh)
- Repeats until full screen drawn
- Same pattern: linear scan of 2D grid

**Advantages**:
- ✅ Simple to implement
- ✅ Deterministic (same order every time)
- ✅ Truly autoregressive (strict causal ordering)
- ✅ No computation needed (just indexing)
- ✅ Easy to batch (all images use same ordering)

**Disadvantages**:
- ❌ **Arbitrary**: No semantic meaning to the order
- ❌ **Doesn't match perception**: Humans don't process images left→right, top→bottom
- ❌ **Ignores structure**: Object boundaries, semantic regions ignored
- ❌ **Long-range dependencies**: Pixel at bottom-right very far from top-left
- ❌ **Slow at pixel level**: 256×256 = 65K sequential predictions!
- ❌ **Context limitations**: Can only use pixels above and to the left (limited receptive field)

**Visual Context Problem**:
```
When predicting pixel X:

✅ Available context:    ❌ Not available yet:
A B C D
E F G X                  ? ? ?
                         ? ? ?
                         ? ? ?

Missing: right side, bottom half of image!
Can't use future context (violates causality)
```

- **Used by**: PixelCNN, PixelRNN, ImageGPT (with patches)
- **Why used**: Simplicity, truly autoregressive
- **Why problematic**: Arbitrary, doesn't match how we perceive images

#### 2. **Coarse-to-Fine Ordering** (Hierarchical)
```
Level 1: Predict 8x8 image
Level 2: Predict 16x16 given 8x8
Level 3: Predict 32x32 given 16x16
...
```
- **Used by**: PixelCNN++, VQ-VAE-2, some diffusion models
- **Pros**:
  - More aligned with perception (gist first, details later)
  - Hierarchical structure
  - More efficient than raster order
- **Cons**:
  - Still somewhat arbitrary
  - Fixed hierarchy may not match semantic importance

#### 3. **Superpixel Sequential Prediction** (Semantic ordering)
```
1. Segment image into superpixels
2. Order superpixels by: saliency, size, position, or learned importance
3. Predict each superpixel given previous ones
```
- **Pros**:
  - More semantic than pixels
  - Flexible ordering strategies
  - Can use SAM for high-quality segments
- **Cons**:
  - How to order superpixels meaningfully?
  - Computationally expensive

**Ordering strategies**:
- **Saliency-based**: Most salient regions first (objects before background)
- **Size-based**: Large regions first, details later
- **Distance-based**: Center→outward or top→bottom
- **Random but consistent**: Fixed random ordering per image
- **Learned ordering**: Model learns optimal prediction order

#### 4. **Video Frame Prediction** (Natural temporal order) ⭐
```
Frame 1 → Frame 2 → Frame 3 → ...
```
- **Used by**: Video prediction models, world models
- **Pros**:
  - **Natural sequential structure** (this is the killer advantage!)
  - Inherent ordering from time
  - Forces understanding of motion, physics, causality
  - Scales perfectly with video data
- **Cons**:
  - Requires video data (but it's abundant!)
  - More compute intensive

**This might be the true image equivalent to next token prediction!**

#### 5. **Patch/Region Autoregressive** (Block-wise)

**Standard Approach: Vision Transformer (ViT) Style Tokenization**

```
1. Patching: Divide image into fixed-size patches (e.g., 16x16 pixels)
2. Flatten: Convert each 2D patch to 1D vector
3. Linear Embedding: Project to fixed-size continuous vector
4. Positional Encoding: Add position information (row, column)
5. Sequence Formation: Arrange in order (typically raster scan)
```

**Process Details**:

1. **Patching**: Image (224×224) → Grid of patches (14×14 patches of 16×16 pixels)
2. **Flattening**: Each patch (16×16×3) → Vector of length 768
3. **Linear Projection**: 768-dim vector → Embedding dimension (e.g., 512-dim)
4. **Positional Encoding**: Add learnable position embeddings to preserve spatial relationships
5. **Raster Scan**: Order patches left→right, top→bottom (positions 0-195)

- **Used by**: Vision Transformers (ViT), Image GPT, Parti, MUSE
- **Pros**:
  - More efficient than pixel-level
  - Can use learned patch embeddings
  - Balances granularity and efficiency
  - Transformer-compatible
  - Enables multimodal integration (vision + text)
- **Cons**:
  - Still needs ordering scheme
  - Patch boundaries may break objects
  - Fixed-size patches may not align with semantic regions

**Ordering options**:
- **Raster** (like iGPT): Left→right, top→bottom (most common)
- **Spiral from center**: Coarse (center) to fine (edges)
- **Random shuffle then predict**: Arbitrary but consistent ordering
- **Hierarchical (quadtree)**: Recursive subdivision
- **Learned ordering**: Model determines optimal sequence

**Impact**:
- **Efficiency**: Compresses image data dramatically (224×224×3 = 150K pixels → 196 tokens)
- **Scalability**: Enables training larger models on more data
- **Multimodal**: Same architecture for vision and language

#### 6. **Latent Space Autoregressive** (Abstract tokens) ⭐

**Vector Quantization Approach: Creating a Visual Vocabulary**

```
Stage 1 - Learn Visual Codebook:
1. Train VQ-VAE/VQGAN encoder-decoder
2. Learn discrete codebook (e.g., 8192 visual "words")
3. Encoder: image → continuous latent → quantize to nearest codebook entry
4. Decoder: discrete code → reconstruct image

Stage 2 - Autoregressive Modeling:
1. Encode image to discrete token IDs (e.g., 32×32 = 1024 tokens)
2. Transformer predicts next token ID given previous tokens
3. Generate images by sampling tokens sequentially
```

**Detailed Pipeline**:

1. **Encode to Latent Space**:
   - Image (256×256×3) → Encoder → Continuous latent (32×32×256)

2. **Vector Quantization**:
   - For each of 1024 spatial positions
   - Find nearest vector in codebook (size 8192)
   - Replace with discrete token ID: 0-8191

3. **Result**:
   - Image → 32×32 grid of discrete token IDs
   - Each token represents a semantic visual concept

4. **Autoregressive Prediction**:
   - Flatten to 1D sequence (1024 tokens in raster order)
   - GPT-style transformer predicts token[i] from token[0:i-1]
   - Can generate images by sampling sequentially

**Models Using This**:
- **VQVAE** (van den Oord, 2017): Original vector quantization approach
- **VQGAN** (Esser, 2021): Improved with adversarial training + perceptual loss
- **DALL-E** (OpenAI, 2021): 8192 codebook, generates images from text
- **Parti** (Google, 2022): ViT-VQGAN tokens, 20B parameter transformer
- **TiTok** (Recent): Highly compressed 1D sequences, variable length
- **FlexTok** (Recent): Adaptive token length based on image complexity

**Pros**:
- **Semantic tokens**: Higher-level than pixels, captures meaningful patterns
- **Extreme efficiency**: 256×256×3 = 196K pixels → 1024 tokens (192× compression)
- **Transformer-compatible**: Can use GPT architecture directly
- **Proven effective**: DALL-E, Parti show this works at scale
- **Discrete vocabulary**: Like words in language (8K-16K "visual words")
- **Generative**: Can sample new images token-by-token

**Cons**:
- **Two-stage training**: Need good encoder first, then transformer
- **Reconstruction quality**: Limited by codebook size and latent resolution
- **Still needs ordering**: Typically raster scan of latent grid
- **Complexity**: More complex than end-to-end approaches

**Compression Comparison**:
- Raw pixels: 256×256×3 = 196,608 values
- ViT patches: 256 tokens (16×16 patches)
- VQ-VAE/VQGAN: 1024 tokens (32×32 latent grid)
- TiTok: ~32-128 tokens (variable, adaptive compression)

---

### 6.5 **Aggressive JPEG Compression as Tokenization** (Practical Fix!) 💡

**Key Insight**: JPEG compression already performs a form of discrete tokenization through its compression pipeline!

**JPEG as Natural Tokenization**:

```
JPEG Compression Pipeline:
1. Color space conversion: RGB → YCbCr
2. Downsampling: Chroma subsampling (4:2:0)
3. Block division: 8×8 pixel blocks
4. DCT (Discrete Cosine Transform): Convert to frequency domain
5. Quantization: Round coefficients → DISCRETE VALUES
6. Entropy coding: Huffman or arithmetic coding

Result: Natural discrete tokens from DCT coefficients!
```

**Why This Works for Sequential Tokens**:

1. **Already Discrete**: Quantized DCT coefficients are integers (discrete tokens)
2. **Natural Ordering**: Zigzag scan provides a meaningful sequence:
   - DC coefficient first (overall brightness)
   - Low-frequency AC coefficients (coarse structure)
   - High-frequency AC coefficients (fine details)
3. **Semantic Hierarchy**: Frequency ordering = coarse-to-fine naturally!
4. **Extreme Compression**: Quality 10-30 JPEG gives 10-50× compression
5. **No Training Needed**: JPEG codec is hand-crafted, fast, universal

**Aggressive JPEG Pipeline**:

```python
# Step 1: Aggressive JPEG compression
image → JPEG(quality=20) → compressed_image

# Step 2: Extract DCT coefficients as tokens
for 8x8_block in image:
    dct_coeffs = DCT(block)  # 64 coefficients per block
    quantized = quantize(dct_coeffs, quality=20)  # Discrete integers
    tokens = zigzag_scan(quantized)  # Order: DC, low-freq → high-freq

# Step 3: Predict tokens autoregressively
for each block_position:
    predict next_64_tokens from previous_blocks

# OR: Predict next block's DC coefficient only (ultra-compressed)
for each block:
    predict next_DC from previous_DCs
```

**Benefits of JPEG Tokenization**:

- **✅ Pre-discretized**: Quantization table already makes tokens discrete
- **✅ Semantic ordering**: Zigzag scan is coarse-to-fine (DC→AC low→AC high)
- **✅ Extreme compression**: Quality 10-30 reduces data massively
- **✅ Fast**: Hardware-accelerated JPEG codecs
- **✅ Universal**: Works on any image
- **✅ Hierarchical**: Frequency domain = natural multi-scale
- **✅ Lossy**: Forces learning of high-level features (can't memorize pixels)

**Aggressive Compression as Regularization**:

- Quality 10-30 JPEG destroys fine details
- Model MUST learn semantic understanding
- Can't rely on pixel-perfect reconstruction
- Similar to aggressive data augmentation
- Forces invariance to compression artifacts

**Practical Implementation**:

```python
import torch
from torchvision import transforms
from PIL import Image
import io

class AggressiveJPEGTokenizer:
    def __init__(self, quality=20):
        self.quality = quality

    def compress(self, image):
        """Aggressive JPEG compression"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=self.quality)
        compressed = Image.open(buffer)
        return compressed

    def extract_dct_tokens(self, image):
        """Extract DCT coefficients as discrete tokens"""
        # Use PIL or libjpeg to access DCT coefficients
        # Each 8×8 block → 64 tokens in zigzag order
        # Return sequence of discrete integer tokens
        pass

    def tokenize(self, image):
        """Full pipeline: compress → extract tokens"""
        compressed = self.compress(image)
        tokens = self.extract_dct_tokens(compressed)
        return tokens
```

**Comparison to VQ-VAE**:

| Aspect | VQ-VAE/VQGAN | Aggressive JPEG |
|--------|--------------|-----------------|
| Training | Required | None (hand-crafted) |
| Speed | Slower (neural net) | Very fast (hardware) |
| Codebook | Learned (8K-16K) | Fixed quantization table |
| Compression | ~200× | 10-50× (configurable) |
| Semantic | Learned features | Frequency domain |
| Ordering | Arbitrary (raster) | Natural (zigzag DC→AC) |

**Hybrid Approach** (Best of both worlds):

```
1. Aggressive JPEG compression (quality 20)
2. Encode JPEG to latent codes with VQ-VAE
3. Result: Extreme compression + learned semantics
```

**Why This is a "Fix"**:

- Solves tokenization without learning
- Provides natural ordering (frequency-based)
- Extreme compression reduces sequence length
- Can bootstrap quickly, iterate with learned tokenizers later
- Aligns with "bootstrapping philosophy" - use existing tools!

**Research Directions**:

1. **JPEG-GPT**: Train GPT directly on JPEG DCT coefficients
2. **Hybrid JPEG-VQ**: JPEG preprocessing + learned quantization
3. **Adaptive quality**: Variable JPEG quality based on image complexity
4. **AC coefficient prediction**: Predict high-freq from low-freq
5. **Multi-quality pyramid**: Stack multiple JPEG quality levels

**Cautionary Notes**:

- JPEG optimized for human perception, not semantic learning
- Block artifacts may not align with object boundaries
- Chroma subsampling loses color information
- Fixed 8×8 blocks may not be optimal for all tasks

**Verdict**: Aggressive JPEG is a practical, fast way to create discrete tokens with natural ordering. Excellent for rapid prototyping before investing in learned tokenizers like VQ-VAE.

---

### 6.6 **Alternatives to JPEG Compression for Tokenization**

JPEG isn't the only compression codec that creates discrete tokens. Here are alternatives with different properties:

#### **1. JPEG 2000** (Wavelet-based)

```
Pipeline:
1. Discrete Wavelet Transform (DWT) instead of DCT
2. Multi-resolution decomposition
3. Quantization → discrete coefficients
4. Better at preserving edges than JPEG
```

**Advantages over JPEG**:
- ✅ Better edge preservation
- ✅ True multi-scale (wavelet pyramid)
- ✅ Less blocking artifacts
- ✅ Perceptually better quality at same compression

**Disadvantages**:
- ❌ Slower than JPEG (less hardware support)
- ❌ More complex
- ❌ Less widespread adoption

**For SSL**: Wavelet coefficients might be more semantically meaningful than DCT. Multi-resolution decomposition aligns with coarse-to-fine learning.

#### **2. WebP** (Google)

```
Pipeline (lossy mode):
1. Similar to VP8 video codec
2. Block-based prediction
3. Transform coding (similar to DCT)
4. Quantization → discrete tokens
```

**Advantages**:
- ✅ Better compression than JPEG (~30% smaller)
- ✅ Fast (hardware accelerated on some platforms)
- ✅ Block-based like JPEG (familiar structure)

**Disadvantages**:
- ❌ Less analyzed for ML applications
- ❌ Codec more complex than JPEG

**For SSL**: Could extract prediction modes and transform coefficients as tokens. May learn better spatial relationships due to intra-prediction.

#### **3. AVIF** (AV1 Image File Format)

```
Pipeline:
1. Based on AV1 video codec
2. Modern transform coding
3. Advanced intra-prediction
4. Very high compression
```

**Advantages**:
- ✅ State-of-the-art compression (50%+ better than JPEG)
- ✅ Modern codec with ML-friendly features
- ✅ Rich prediction modes

**Disadvantages**:
- ❌ Slow encoding (not hardware accelerated everywhere)
- ❌ Complex codec

**For SSL**: Prediction modes could be valuable tokens. Learns spatial relationships during encoding.

#### **4. HEIC/HEIF** (Apple/MPEG)

```
Pipeline:
1. Based on H.265/HEVC video codec
2. Intra-frame prediction
3. Transform + quantization
4. Used by Apple Photos
```

**Advantages**:
- ✅ Good compression
- ✅ Hardware support on Apple devices
- ✅ Video codec technology

**Disadvantages**:
- ❌ Patent encumbered
- ❌ Platform specific
- ❌ Complex

#### **5. Video Codecs as Image Compressors**

**H.264 Intra Frames**:
```
Use I-frames from H.264 video:
- Spatial prediction (9 intra modes)
- 4×4 or 8×8 integer transform
- Quantization → discrete tokens
- Natural sequential ordering from prediction dependencies
```

**Advantages**:
- ✅ Extremely optimized (hardware everywhere)
- ✅ Spatial prediction modes = learned relationships
- ✅ Natural ordering from prediction dependencies
- ✅ Can extend to inter-frame prediction (video)

**H.265/HEVC, AV1**:
- More prediction modes (35+ for HEVC)
- Better compression
- Richer token vocabulary

**For SSL**: Video codec prediction modes might teach better spatial relationships than JPEG. Natural path to video prediction.

#### **6. Wavelet Transforms** (General)

**Discrete Wavelet Transform (DWT)**:
```
1. Decompose into wavelets at multiple scales
2. Coefficient hierarchy: LL, LH, HL, HH per level
3. Quantize coefficients → discrete tokens
4. Natural coarse-to-fine ordering
```

**Types**:
- Haar wavelets (simplest)
- Daubechies wavelets
- Biorthogonal wavelets

**Advantages**:
- ✅ True multi-resolution
- ✅ Better edge preservation than DCT
- ✅ Natural hierarchical structure
- ✅ Mathematically cleaner than DCT

**Disadvantages**:
- ❌ No standard image format (need custom)
- ❌ Less hardware support

**For SSL**: Excellent for multi-scale learning. Coarse-to-fine is explicit in the decomposition.

#### **7. Learned Compression (Neural Codecs)**

**Variational Autoencoders + Entropy Models**:
```
1. Neural encoder: image → latent
2. Quantization
3. Entropy model (context-adaptive)
4. Neural decoder: latent → image
```

**Examples**:
- Ballé et al. neural compression
- Learned Image Compression (LIC)
- VVC (Versatile Video Coding) with neural tools

**Advantages**:
- ✅ Can learn compression specifically for task
- ✅ State-of-the-art rate-distortion
- ✅ Flexible latent structure

**Disadvantages**:
- ❌ Requires training the codec itself
- ❌ Slower than traditional codecs
- ❌ Defeats purpose of "zero training" bootstrap

**For SSL**: Best long-term but defeats quick prototyping. Similar to VQ-VAE approach.

#### **8. Other Transform Domains**

**Fourier Transform**:
```
- 2D FFT
- Magnitude and phase
- Low freq → high freq ordering
```
- Similar to DCT but whole-image
- Less localized than DCT/wavelets

**Hadamard Transform**:
```
- Integer-only transform
- Very fast (no multiplications)
- Used in some video codecs
```
- Simpler than DCT
- Less compression but faster

#### **Comparison Table: Compression Methods for Tokenization**

| Method | Ordering Quality | Compression | Speed | Hardware | Complexity | Bootstrap Time |
|--------|-----------------|-------------|-------|----------|------------|----------------|
| **JPEG (DCT)** | ⭐⭐⭐⭐ Zigzag | Good | ⭐⭐⭐⭐⭐ | ✅ Yes | Low | Minutes |
| **JPEG 2000 (DWT)** | ⭐⭐⭐⭐⭐ Multi-scale | Better | ⭐⭐⭐ | ⚠️ Partial | Medium | Minutes |
| **WebP** | ⭐⭐⭐ Prediction | Better | ⭐⭐⭐⭐ | ⚠️ Partial | Medium | Minutes |
| **AVIF** | ⭐⭐⭐⭐ Prediction | Best | ⭐⭐ | ❌ No | High | Minutes |
| **H.264 I-frames** | ⭐⭐⭐⭐⭐ Prediction | Good | ⭐⭐⭐⭐⭐ | ✅ Yes | Medium | Minutes |
| **Wavelets (custom)** | ⭐⭐⭐⭐⭐ Hierarchy | Variable | ⭐⭐⭐ | ❌ No | Medium | Hours (coding) |
| **Neural Codec** | ⭐⭐⭐⭐⭐ Learned | Best | ⭐ | ❌ No | Very High | Weeks (training) |

#### **Recommendations by Use Case**:

1. **Quick Prototyping** (Days):
   - **JPEG**: Best balance of speed, simplicity, hardware support
   - **H.264 I-frames**: If you want prediction modes

2. **Better Semantics** (Days-Week):
   - **JPEG 2000**: Wavelet multi-scale, better edges
   - **Wavelets (custom)**: Full control over decomposition

3. **Path to Video** (Weeks):
   - **H.264/H.265**: Start with I-frames, extend to P/B frames
   - Natural progression to temporal prediction

4. **Maximum Compression** (Weeks):
   - **AVIF/AV1**: State-of-the-art, rich prediction modes
   - **Neural codecs**: If willing to train

5. **Research Exploration** (Ongoing):
   - Try multiple: JPEG vs JPEG2000 vs Wavelet vs H.264
   - Compare which learns better representations
   - Ablation study on transform type

#### **Hybrid Approaches**:

**Multi-codec ensemble**:
```
1. Extract tokens from multiple codecs
2. JPEG (DCT) + JPEG2000 (Wavelet) + H.264 (Prediction)
3. Learn which codec tokens are most informative
4. Ensemble or select per image region
```

**Progressive refinement**:
```
1. Start with JPEG quality 10 (coarse)
2. Progressively add quality 20, 40, 80
3. Learn to predict refinement deltas
4. Natural curriculum from coarse to fine
```

#### **Key Insight**:

The choice of compression codec determines:
- **Token vocabulary** (DCT vs wavelet vs prediction modes)
- **Ordering** (zigzag vs hierarchical vs prediction order)
- **Semantics** (frequency vs spatial vs hybrid)

JPEG is the practical default, but JPEG 2000 (wavelets) or H.264 I-frames (prediction) might learn better spatial relationships. Experimentation needed!

**Verdict**: Aggressive JPEG is a practical, fast way to create discrete tokens with natural ordering. Excellent for rapid prototyping before investing in learned tokenizers like VQ-VAE.

#### 7. **Gradient-Based Adaptive Walk** (Content-aware traversal) 🔥 NEW!

**Key Insight**: Let image content determine the traversal order through gradient-based navigation!

**Concept: Visual Gradient Walk**

```python
# Walk that follows gradients (most change)
current_position = start_point
predicted_sequence = []

while not_all_visited:
    # Predict current pixel/patch
    prediction = predict(current_position, context=predicted_sequence)
    predicted_sequence.append(prediction)

    # Move in direction of maximum gradient
    gradient_map = compute_gradients(image)
    next_position = argmax(gradient_map[neighbors(current_position)])
    current_position = next_position
```

**Two Variants**:

**A) Maximum Gradient Walk** (Follow edges/boundaries):
```
Start → Move to neighbor with highest gradient magnitude
Result: Walk follows edges, object boundaries, salient features
Order: Background → Edges → Details
```

**B) Minimum Gradient Walk** (Follow smooth regions):
```
Start → Move to neighbor with lowest gradient magnitude
Result: Walk follows homogeneous regions first
Order: Smooth areas → Gradual transitions → Edges
```

**Why This is Powerful**:

1. **Content-Aware**: Ordering adapts to actual image structure
2. **Mimics Human Vision**: Similar to saccadic eye movements (we look at high-contrast regions)
3. **Semantic**: Edges/boundaries often correspond to object boundaries
4. **Natural**: Follows perceptual saliency
5. **Learnable**: Model can learn optimal gradient-following policy

**Implementation Strategies**:

**Option 1: Fixed Gradient Walk** (Hand-crafted)
```python
def gradient_walk_ordering(image):
    """Create pixel ordering by following gradients"""
    # Compute image gradients (Sobel, Scharr, etc.)
    grad_x = sobel_x(image)
    grad_y = sobel_y(image)
    gradient_magnitude = sqrt(grad_x**2 + grad_y**2)

    # Start from center (or lowest gradient point)
    current = image_center
    visited = set()
    ordering = []

    while len(visited) < num_pixels:
        ordering.append(current)
        visited.add(current)

        # Get unvisited neighbors
        neighbors = get_neighbors(current, visited)

        # Move to neighbor with max gradient (edge-following)
        next_pos = max(neighbors, key=lambda p: gradient_magnitude[p])
        current = next_pos

    return ordering
```

**Option 2: Learned Walk Policy** (Reinforcement Learning)
```python
def learned_walk_policy(image, policy_network):
    """Learn optimal walk policy for prediction"""
    current = start_position
    ordering = []

    while not_done:
        # Policy network decides where to go next
        # Based on: current position, image features, prediction difficulty
        features = extract_features(image, current, ordering)
        next_direction = policy_network(features)

        current = move(current, next_direction)
        ordering.append(current)

    return ordering
```

**Option 3: Multi-Scale Gradient Walk**
```python
# Walk at multiple scales simultaneously
coarse_walk = gradient_walk(downsample(image, 8x))  # 8x8 regions
medium_walk = gradient_walk(downsample(image, 4x))  # 4x4 regions
fine_walk = gradient_walk(image)                    # Pixels

# Hierarchical: Coarse regions → Medium → Fine within each region
```

**Gradient Computation Options**:

1. **Spatial Gradients** (Standard):
   - Sobel, Scharr, Prewitt operators
   - Edge magnitude and direction

2. **Perceptual Gradients**:
   - Difference in learned features (CNN layer activations)
   - Semantic boundaries rather than just intensity

3. **Prediction Difficulty Gradients**:
   - Estimate how hard each region is to predict
   - Walk from easy → hard or hard → easy

4. **Saliency-Based Gradients**:
   - Use saliency maps (attention models)
   - Walk follows visual importance

**Analogies to Human Vision**:

```
Human Eye Movements:
1. Saccades: Jump to salient regions (high gradient)
2. Fixations: Predict/process local region
3. Smooth pursuit: Follow moving edges

Our Gradient Walk:
1. Move to high-gradient regions (edges, objects)
2. Predict at current position
3. Continue along boundaries/features
```

**Advantages**:

- ✅ **Content-aware ordering**: Not arbitrary like raster
- ✅ **Semantic**: Follows object boundaries naturally
- ✅ **Adaptive**: Different path for each image
- ✅ **Perceptually motivated**: Mimics visual attention
- ✅ **Edge-aware**: Natural for object-centric learning
- ✅ **Can be learned**: Policy network optimizes for predictability

**Challenges**:

- ❌ **Non-deterministic**: Different ordering per image (harder to batch?)
- ❌ **Connectivity**: How to handle disconnected regions?
- ❌ **Starting point**: Where to begin the walk?
- ❌ **Backtracking**: What if walk gets stuck?
- ❌ **Computational cost**: Need to compute gradients first

**Advanced Variants**:

**1. Bidirectional Gradient Walk**:
```
Two simultaneous walks:
- Walk A: Follows maximum gradients (edges)
- Walk B: Follows minimum gradients (smooth regions)
- Predict: Can one walk predict the other's path?
```

**2. Gradient-Guided Superpixel Ordering**:
```
1. Segment into superpixels (SAM)
2. Compute gradient between adjacent superpixels
3. Walk through superpixels following boundary strength
4. Predict each superpixel in traversal order
```

**3. Curriculum: Smooth → Edges**:
```
Epoch 1: Walk follows minimum gradients (easy, smooth regions)
Epoch 2: Walk follows medium gradients (moderate difficulty)
Epoch 3: Walk follows maximum gradients (hard, edges/details)

Progressive difficulty like coarse-to-fine
```

**4. Multi-Agent Walks**:
```
Launch multiple walkers from different starting points:
- Walker 1: From top-left, follows edges
- Walker 2: From center, follows saliency
- Walker 3: From brightest region, follows brightness
- Predict: When/where will walkers meet?
```

**Research Questions**:

1. **Max vs Min Gradients**: Which learns better representations?
   - Max: Focuses on edges, boundaries (harder to predict)
   - Min: Focuses on smooth regions (easier to predict)

2. **Starting Point**: Where to begin?
   - Center (saliency-weighted center)
   - Random (different each time)
   - Lowest gradient (easiest first)
   - Highest gradient (hardest first)

3. **Walk Strategy**:
   - Greedy (always max gradient)
   - Epsilon-greedy (explore vs exploit)
   - Learned policy (RL agent)

4. **Prediction Task**:
   - Predict pixel/patch values
   - Predict gradient magnitude at next position
   - Predict optimal next direction
   - Predict full remaining path

**Comparison to Other Methods**:

| Aspect | Raster | Coarse-Fine | Gradient Walk |
|--------|--------|-------------|---------------|
| Ordering | Fixed | Fixed | Adaptive |
| Content-aware | ❌ No | ⚠️ Partial | ✅ Yes |
| Per-image | Same | Same | Unique |
| Semantic | ❌ No | ⚠️ Scale | ✅ Boundaries |
| Complexity | Low | Medium | High |

**Practical Implementation** (Hybrid approach):

```python
class GradientWalkTokenizer:
    def __init__(self, walk_type='max_gradient'):
        self.walk_type = walk_type

    def compute_ordering(self, image):
        """Determine pixel ordering via gradient walk"""
        if self.walk_type == 'max_gradient':
            return self.max_gradient_walk(image)
        elif self.walk_type == 'min_gradient':
            return self.min_gradient_walk(image)
        else:
            return self.learned_walk(image)

    def max_gradient_walk(self, image):
        gradients = compute_sobel(image)
        # Start from center
        current = (H//2, W//2)
        ordering = []
        visited = set()

        while len(visited) < H * W:
            ordering.append(current)
            visited.add(current)

            # 8-connected neighbors
            neighbors = [(current[0]+dy, current[1]+dx)
                        for dy, dx in [(-1,-1),(-1,0),(-1,1),
                                       (0,-1),(0,1),
                                       (1,-1),(1,0),(1,1)]]
            neighbors = [n for n in neighbors if n not in visited]

            if not neighbors:
                # Stuck! Jump to unvisited pixel with max gradient
                unvisited = [(i,j) for i in range(H) for j in range(W)
                           if (i,j) not in visited]
                current = max(unvisited, key=lambda p: gradients[p])
            else:
                # Move to neighbor with max gradient
                current = max(neighbors, key=lambda p: gradients[p])

        return ordering

    def tokenize(self, image):
        """Convert image to sequence via gradient walk"""
        ordering = self.compute_ordering(image)
        # Extract pixels/patches in walk order
        return [image[pos] for pos in ordering]
```

**Combining with Video**:

```python
# Temporal + Spatial gradient walk
for frame in video:
    # Spatial walk within frame
    spatial_ordering = gradient_walk(frame)

    # Predict along spatial walk
    for position in spatial_ordering:
        predict(position | previous_positions + previous_frames)

    # Temporal walk: Track high-gradient regions across frames
    # Edges moving through time = object motion
```

**Why This Could Be VERY Powerful**:

1. **Learning object boundaries**: Max gradient walk naturally follows edges
2. **Saliency-based**: Mimics where humans look
3. **Adaptive**: Each image gets optimal traversal
4. **Compositional**: Can combine spatial gradient walk + temporal prediction
5. **Learnable policy**: Can optimize walk strategy for downstream tasks

**Verdict**: Gradient-based walks are a novel, perceptually-motivated approach to creating content-aware sequential orderings. Could be superior to arbitrary raster ordering. Worthy of research exploration!

#### 8. **Attention-Based Dynamic Ordering** (Let model decide)
```
Model learns which regions to predict next based on context
```
- **Used by**: Some non-autoregressive models, diffusion models
- **Pros**:
  - Flexible, adaptive to image content
  - Could discover optimal ordering
- **Cons**:
  - Not truly autoregressive
  - More complex training

#### 8. **Masked Token Prediction** (BERT-style, not strictly autoregressive)
```
Randomly mask tokens, predict them from unmasked context
```
- **Used by**: MAE, BEiT, SimMIM
- **Pros**:
  - Very effective in practice
  - Bidirectional context (better than pure autoregressive?)
  - Efficient training
- **Cons**:
  - Not autoregressive (can't generate sequentially)
  - Less like "next token prediction"

### Comparative Analysis

| Approach | Sequential? | Semantic? | Efficient? | Natural Order? | Like Next Token? | Training Cost | Content-Aware? |
|----------|-------------|-----------|------------|----------------|------------------|---------------|----------------|
| Raster order | ✅ Yes | ❌ No | ❌ Slow | ❌ No | ⭐⭐ Somewhat | Medium | ❌ No |
| Coarse-to-fine | ✅ Yes | ⚠️ Partial | ✅ Better | ⚠️ Partial | ⭐⭐⭐ Good | Medium | ❌ No |
| Superpixel sequence | ✅ Yes | ✅ Yes | ⚠️ Medium | ⚠️ Depends | ⭐⭐⭐ Good | High | ⚠️ Partial |
| **Video frames** | ✅ **Yes** | ✅ **Yes** | ✅ **Good** | ✅ **Yes!** | ⭐⭐⭐⭐⭐ **Best!** | High | ⚠️ **Temporal** |
| Patch autoregressive | ✅ Yes | ⚠️ Partial | ✅ Good | ❌ No | ⭐⭐⭐ Good | Medium | ❌ No |
| Latent autoregressive | ✅ Yes | ✅ Yes | ✅ Good | ❌ No | ⭐⭐⭐⭐ Great | Very High | ❌ No |
| **Aggressive JPEG** | ✅ **Yes** | ⚠️ **Freq** | ✅ **Great** | ✅ **Yes!** | ⭐⭐⭐⭐ **Great** | **None!** | ❌ No |
| **Gradient walk** 🔥 | ✅ **Yes** | ✅ **Yes!** | ⚠️ **Medium** | ✅ **Yes!** | ⭐⭐⭐⭐ **Great** | Low-Medium | ✅ **Yes!** |
| Masked prediction | ❌ No | ✅ Yes | ✅ Great | N/A | ⭐⭐ Different | Medium | ❌ No |

**Key Advantages by Approach**:
- **Video frames**: Most natural, temporal causality, but needs video data
- **Gradient walk**: Content-aware, mimics human vision, perceptually motivated! NEW!
- **Latent autoregressive**: Semantic tokens, proven at scale, but two-stage training
- **Aggressive JPEG**: Zero training, natural frequency ordering, fast prototyping
- **Masked prediction**: Not truly sequential but very effective in practice

---

## Beyond 1D: Truly N-Dimensional Autoregressive Prediction 🔥 NEW!

### The Fundamental Question

**All approaches above linearize 2D images into 1D sequences. But why?**

This is a limitation inherited from language models, where text is naturally 1D. But images are inherently 2D (or 3D with time/depth). **Can we do "next token prediction" in a truly N-dimensional way?**

### The Core Insight: Frontier Prediction vs. Sequence Prediction

#### 1D Next Token Prediction (Language)
```
Context:  [t-k, ..., t-2, t-1]
Predict:  t

Sequential: Only one "next" token exists
```

#### 2D Frontier Prediction (Images)
```
Context:  Filled region R ⊂ Z^2
Predict:  Frontier ∂R (boundary pixels)

Parallel: Multiple frontier pixels can be predicted simultaneously
Spatial: Context is 2D neighborhood, not 1D history
```

#### N-D Frontier Prediction (General)
```
Context:  Filled region R ⊂ Z^N
Predict:  Frontier ∂R (N-1 dimensional boundary)

Examples:
- 2D images: Predict boundary pixels given interior
- 3D volumes: Predict surface voxels given interior
- Video (2D+time): Predict spatial boundary at next frame
```

### Why This Matters

**Linearization destroys spatial structure:**
- Raster scan: Pixel (0,1) is "before" (1,0), but spatially they're neighbors
- Walk-based: Arbitrary ordering based on heuristic
- Sequential models: Process non-adjacent spatial neighbors at very different timesteps

**Frontier prediction preserves spatial structure:**
- Context is naturally 2D/ND: The filled region is a spatial set
- Multiple predictions in parallel: All frontier pixels predicted together
- Spatial coherence: Nearby frontier pixels share nearby context
- Natural for images: Growing a region is more natural than scanning

### Extending Transformers to N-Dimensions

#### Challenge: Transformers Expect Sequences

Standard transformer:
```python
# Input: (batch, sequence_length, d_model)
# Attention: Each position attends to all previous positions
# Output: (batch, sequence_length, d_model)

# Position i can only attend to positions [0, 1, ..., i-1]
# This is 1D causality
```

#### Solution 1: Spatial Masked Attention (PixelCNN-style)

Generalize causality to N dimensions:

```python
# 2D Example: Raster order causality
# Pixel (r, c) can attend to:
#   - All pixels in rows [0, ..., r-1]
#   - All pixels in row r with columns [0, ..., c-1]

class SpatialMaskedAttention(nn.Module):
    """
    N-dimensional causal attention.

    For 2D images with raster ordering:
        Position (r1, c1) can attend to (r2, c2) iff:
            - r2 < r1, OR
            - r2 == r1 and c2 < c1
    """

    def create_2d_causal_mask(self, H, W):
        # Flatten 2D positions to 1D
        positions = torch.arange(H * W).view(H, W)

        # Create mask based on 2D causality
        mask = torch.zeros(H * W, H * W)
        for r1 in range(H):
            for c1 in range(W):
                for r2 in range(H):
                    for c2 in range(W):
                        idx1 = r1 * W + c1
                        idx2 = r2 * W + c2

                        # Can attend if (r2, c2) is "before" (r1, c1)
                        if r2 < r1 or (r2 == r1 and c2 < c1):
                            mask[idx1, idx2] = 1

        return mask
```

**Key insight**: This maintains 1D sequence but enforces 2D causality in the attention mask.

#### Solution 2: Coordinate-Based Transformers (N-D Native)

Inspired by NeRF, Perceiver, and coordinate-based networks:

```python
class CoordinateTransformer(nn.Module):
    """
    Truly N-dimensional transformer.

    Instead of position indices, use actual N-D coordinates.
    """

    def __init__(self, n_dims, d_model, n_heads):
        self.n_dims = n_dims
        self.coord_encoder = FourierFeatures(n_dims, d_model // 2)
        self.transformer = Transformer(d_model, n_heads)

    def forward(self, coords, values, query_coords):
        """
        Args:
            coords: (batch, n_context, n_dims) - Coordinates of known values
            values: (batch, n_context, d_value) - Values at those coordinates
            query_coords: (batch, n_query, n_dims) - Coordinates to predict

        Returns:
            (batch, n_query, d_value) - Predicted values at query coordinates
        """
        # Encode coordinates with Fourier features
        context_encoding = self.coord_encoder(coords)  # (batch, n_context, d_model)
        query_encoding = self.coord_encoder(query_coords)  # (batch, n_query, d_model)

        # Combine coordinate encoding with values
        context = torch.cat([context_encoding, values], dim=-1)

        # Attend from queries to context
        # No sequence ordering - purely spatial relationships
        output = self.transformer(query_encoding, context)

        return output


class FourierFeatures(nn.Module):
    """
    Map N-D coordinates to high-frequency features.
    From NeRF and coordinate-based networks.
    """

    def forward(self, coords):
        # coords: (..., n_dims)
        # Map to Fourier features: [sin(2πBx), cos(2πBx)]
        # where B is a random frequency matrix

        freq = coords @ self.B  # (..., d_model // 2)
        return torch.cat([torch.sin(freq), torch.cos(freq)], dim=-1)
```

**Key insight**: No linearization! Coordinates stay as N-D vectors. Attention is purely spatial, not sequential.

#### Solution 3: Frontier-Based Autoregressive

Explicitly model the growing frontier:

```python
class FrontierPrediction(nn.Module):
    """
    Predict values at the frontier given filled region.

    For images:
        1. Start with seed region (e.g., center pixel)
        2. Identify frontier (unvisited neighbors of filled region)
        3. Predict all frontier pixels in parallel
        4. Add predictions to filled region
        5. Repeat until image complete
    """

    def forward(self, image_shape):
        H, W = image_shape
        filled = set()
        predictions = {}

        # Initialize: Start from center
        center = (H // 2, W // 2)
        filled.add(center)
        predictions[center] = self.seed_value

        while len(filled) < H * W:
            # Find frontier
            frontier = self.get_frontier(filled, H, W)

            if not frontier:
                break  # No more frontier (disconnected regions)

            # Gather context for each frontier pixel
            contexts = []
            for (r, c) in frontier:
                # Context: All filled neighbors
                neighbors = self.get_filled_neighbors((r, c), filled)
                context = self.build_context(neighbors, predictions)
                contexts.append(context)

            # Predict all frontier pixels in parallel
            frontier_values = self.predict(contexts)  # (n_frontier, d_value)

            # Add to filled region
            for i, (r, c) in enumerate(frontier):
                filled.add((r, c))
                predictions[(r, c)] = frontier_values[i]

        return predictions

    def get_frontier(self, filled, H, W):
        """Get all unvisited neighbors of filled region."""
        frontier = set()
        for (r, c) in filled:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4-connected
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in filled:
                    frontier.add((nr, nc))
        return list(frontier)
```

**Key insight**: Predict multiple pixels in parallel at each step. More efficient and preserves spatial structure.

### Comparison: 1D vs. N-D Prediction

| Aspect | 1D Sequential | N-D Frontier |
|--------|---------------|--------------|
| **Ordering** | Must linearize | No linearization needed |
| **Context** | Previous tokens in sequence | Spatial neighborhood (filled region) |
| **Parallelization** | Sequential (slow) | Parallel frontier (faster) |
| **Spatial coherence** | Destroyed by linearization | Preserved naturally |
| **Computational** | O(n²) attention for sequence length n | O(f²) where f = frontier size << n |
| **Causality** | 1D: t depends on [1..t-1] | N-D: position depends on spatial neighbors |
| **Natural for** | Text, audio, time series | Images, volumes, spatial data |

### Practical Implementation Strategies

#### Strategy 1: Hybrid (Best of Both Worlds)

Use 1D transformers but with N-D awareness:

```python
class Hybrid2DTransformer(nn.Module):
    """
    1D transformer backbone with 2D positional encodings and attention masks.
    """

    def __init__(self, H, W, d_model):
        self.pos_embed_2d = nn.Parameter(torch.randn(1, H, W, d_model))
        self.transformer = Transformer(d_model)

    def forward(self, x):
        # x: (batch, H, W, d_model)

        # Add 2D positional embeddings
        x = x + self.pos_embed_2d

        # Flatten to sequence but keep 2D structure in attention mask
        batch, H, W, d = x.shape
        x_flat = x.view(batch, H * W, d)

        # Create 2D causal mask
        mask = self.create_2d_causal_mask(H, W)

        # Apply transformer with 2D awareness
        output = self.transformer(x_flat, mask=mask)

        # Reshape back to 2D
        return output.view(batch, H, W, d)
```

#### Strategy 2: Pure N-D (Research Frontier)

No sequences at all - pure coordinate-based:

```python
# Sample random query coordinates
query_coords = torch.rand(batch, n_query, 2)  # Random (x, y) positions

# Context: Previously "filled" coordinates
context_coords = get_filled_coordinates()  # (batch, n_context, 2)
context_values = get_filled_values()        # (batch, n_context, d_value)

# Predict at query coordinates
predictions = model(context_coords, context_values, query_coords)
```

This is completely dimension-agnostic and scales to any N!

### Connection to Existing Methods

**PixelCNN**: 2D causal convolutions ≈ local spatial masked attention

**Image Transformer**: Uses 2D positional encodings but flattens to 1D sequence

**Perceiver**: Coordinate-based, can handle arbitrary input dimensions

**NUWA/Make-A-Video**: Extends to 3D (2D space + time) with factorized attention

**NeRF/Coordinate Networks**: Pure N-D, no sequences, coordinate-based prediction

### Why This Hasn't Been Standard

1. **Transformers were designed for 1D** (language), and linearization worked well enough
2. **Convolutions handled 2D** better than early transformers (inductive bias)
3. **Computational cost**: Full N-D attention is expensive
4. **Success of masked prediction** (MAE, BEiT) reduced need for autoregressive

### The Future: True N-D Foundation Models?

**Key Question**: Can we build foundation models that are natively N-dimensional?

Imagine:
- **Same architecture** for images, videos, 3D shapes, point clouds
- **Coordinate-based**: Just different N-D input coordinates
- **No linearization artifacts**
- **Natural for spatial data**

```python
# Universal N-D Foundation Model
model = UniversalNDModel(d_model=768)

# Use for 2D images
predictions_2d = model(coords_2d, values_2d, query_coords_2d)

# Use for 3D volumes
predictions_3d = model(coords_3d, values_3d, query_coords_3d)

# Use for video (2D + time)
predictions_video = model(coords_2dt, values_2dt, query_coords_2dt)

# Use for point clouds
predictions_points = model(coords_xyz, values_xyz, query_coords_xyz)
```

This is the true generalization of "next token prediction" to N dimensions!

### Practical Next Steps

For this project, we could implement:

1. **2D Frontier Prediction Dataset**: Create training data from our image walks
2. **Spatial Masked Attention**: Extend standard transformers with 2D causal masks
3. **Coordinate Transformer**: Implement pure coordinate-based prediction
4. **Hybrid Model**: Best of both worlds - transformer backbone with N-D awareness

See `utils/nd_prediction.py` for initial implementation.

---

### Recommendations: Practical Roadmap

#### **Tier 1A: Quick Start with Aggressive JPEG** 💡 (Days to implement)

**Best for rapid prototyping and validation:**

```python
# Zero training required!
1. Aggressive JPEG compression (quality 10-30)
2. Extract DCT coefficients as discrete tokens
3. Train transformer on zigzag-ordered coefficients
4. Natural coarse-to-fine ordering (DC → low-freq → high-freq)
```

**Why start here**:
- ✅ No encoder training needed (bootstrap immediately)
- ✅ Natural frequency-based ordering
- ✅ Hardware-accelerated compression
- ✅ Validates "next token prediction" approach quickly
- ✅ Can iterate to learned tokenizers later

#### **Tier 1B: Gradient Walk Exploration** 🔥 (Days-Week to implement)

**Novel content-aware approach:**

```python
# Content-driven ordering!
1. Compute image gradients (Sobel)
2. Walk through image following max/min gradients
3. Predict tokens in walk order
4. Learns edge-following, mimics human visual attention
```

**Why explore this**:
- ✅ Content-aware ordering (not arbitrary!)
- ✅ Mimics saccadic eye movements
- ✅ Naturally follows object boundaries
- ✅ Can combine with superpixels (gradient-guided SAM)
- ✅ Low training cost (simple gradient computation)
- ✅ Could be superior to raster/fixed orderings

#### **Tier 2: Best Long-term - Multi-Scale Video Frame Prediction** ⭐⭐⭐⭐⭐ (Weeks-Months)

**True analogy to next token prediction:**

```python
# Temporal + Hierarchical
for frame in video:
    # Coarse-to-fine per frame
    predict_8x8_features(frame | previous_frames)
    predict_16x16_features(frame | previous_frames + 8x8)
    predict_32x32_features(frame | previous_frames + 16x16)
    # ...
```

**Why this is optimal**:
1. ✅ Natural sequential order (time) - THE KEY ADVANTAGE
2. ✅ Hierarchical structure (coarse-to-fine)
3. ✅ Semantic understanding required
4. ✅ Scalable with data
5. ✅ Autoregressive like language
6. ✅ Forces learning of dynamics, physics, causality

**Best for**: Final production system, research, when video data available

#### **Tier 3: Proven Alternative - Latent Code Autoregressive** ⭐⭐⭐⭐ (Weeks)

**For static images, proven at scale:**

```python
# 1. Learn discrete codebook (VQ-VAE/VQGAN)
image → encoder → discrete_codes (e.g., 32x32 tokens)

# 2. Transformer predicts codes autoregressively
for position in range(32*32):
    predict next_code from previous_codes

# 3. Decode to image
codes → decoder → image
```

**Why this works**:
- High-level semantic tokens (not pixels)
- Can use transformer architecture (like GPT)
- Proven effective (DALL-E, Parti, TiTok)
- Two-stage: train encoder first, then transformer

**Best for**: Static images, when you have compute for two-stage training

#### **Recommended Development Path**:

```
Phase 1 (Week 1):
  → Aggressive JPEG tokenization
  → Validate autoregressive prediction works
  → Fast iteration on architecture

Phase 2 (Weeks 2-4):
  → Train VQ-VAE/VQGAN encoder
  → Compare learned tokens vs JPEG tokens
  → Hybrid approach: JPEG + VQ

Phase 3 (Months):
  → Scale to video frame prediction
  → Multi-scale temporal modeling
  → Full autoregressive world model
```

## Image-Based Tasks

### 1. Superpixel-Based Masking (HIGHLY RECOMMENDED)
**Confidence: 8/10 from model consensus**

- **Irregular region prediction**: Mask out superpixels instead of square patches
  - More aligned with object boundaries and human perception
  - Uses SLIC algorithm for segmentation
  - Forces model to understand semantic grouping

- **Multi-scale superpixel prediction**:
  - Generate superpixels at different granularities (coarse to fine)
  - Learn hierarchical representations
  - Capture both local details and global structure

- **Benefits**: Object-centric representations, better boundary awareness, perceptual alignment with Gestalt principles
- **Challenges**: Computational overhead, potential overfitting to segmentation artifacts
- **Implementation**: 2-4 weeks for proof-of-concept

### 2. Spatial Reasoning Tasks

**Predict nearby/adjacent objects**:
- Given a region, predict what objects are likely to appear nearby
- Learn spatial relationships and scene context
- Can be formulated as multi-label classification or embedding distance task

**Spatial arrangement prediction**:
- Predict the relative positions of image regions
- 8-way spatial relationship classification (above, below, left, right, etc.)
- Forces understanding of object layouts

**Object completion**:
- Given partial object views, complete the full object
- Learn object structure and coherence

### 2.5 Context Reasoning Tasks (NEW)

**Out-of-context object detection** (HIGHLY PROMISING):
- Given an image, identify which object doesn't belong in the scene
- Example: A penguin in a desert, a surfboard in an office
- Forces deep understanding of semantic scene context and co-occurrence patterns

**Implementation approaches**:
- **Synthetic method**: Paste random objects into scenes, model predicts anomaly
- **Self-supervised method**: Learn scene-object co-occurrence from natural images, detect violations
- **Contrastive method**: Objects that appear together vs. objects that don't

**Benefits**:
- Requires high-level semantic understanding
- Can't be solved with low-level features alone
- Naturally handles long-tail and rare combinations
- Relevant for safety-critical applications (anomaly detection in autonomous driving)

**Context consistency prediction**:
- Given multiple objects in a scene, predict if they form a coherent context
- Binary classification: consistent vs. inconsistent scene
- Learn what "makes sense" visually

**Object-scene compatibility**:
- Predict compatibility score between object and background scene
- Continuous score rather than binary
- More nuanced than simple outlier detection

**Missing context prediction**:
- Given a scene, predict what objects are likely missing
- "This is a kitchen, but there's no refrigerator"
- Learn expected object co-occurrences

### 2.75 Transformation and Operation Learning (NEW)

**Operation prediction from input-output pairs**:
- Given original image and transformed image, predict the operation applied
- Examples: rotation (0°, 90°, 180°, 270°), flip, blur, color shift, crop, scale
- Forces understanding of image transformations and invariances

**Implementation approaches**:
- **Classification**: Predict operation class from discrete set
- **Regression**: Predict continuous parameters (rotation angle, blur sigma, scale factor)
- **Sequence prediction**: For compositions of operations (rotate → blur → crop)

**Benefits**:
- Learns what changes vs. what's invariant under transformations
- Meta-learning about image operations
- Can help with understanding augmentation strategies
- Useful for learning disentangled representations

**Operation sequence prediction**:
- Given sequence: Image A → B → C, predict the operations
- Learn compositional transformations
- More challenging than single operation

**Inverse operation prediction**:
- Given transformed image, predict parameters to undo the transformation
- Learn inverse mappings
- Useful for image restoration tasks

**Operation consistency**:
- Apply operations in different orders, check if model understands commutativity
- Example: Rotate then flip vs. flip then rotate
- Learn algebraic structure of transformations

### 3. Semantic Grouping Tasks

**Perceptual grouping (Gestalt principles)**:
- Predict which regions belong to the same perceptual group
- Based on proximity, similarity, continuity, closure
- Aligns with human visual processing

**Figure-ground separation**:
- Predict which regions are foreground vs background
- Can use depth cues, occlusion, texture

**Boundary detection**:
- Predict object boundaries without labels
- Use superpixel edges, color gradients, texture discontinuities

### 4. Affordance and Interaction

**Affordance prediction** (from Grok-4 suggestion):
- Predict how objects can be interacted with
- "Graspable", "sittable", "pushable" regions
- More semantic than pure visual features
- Challenge: Hard to self-supervise without implicit labels

**Physical property prediction**:
- Predict material properties (rough, smooth, rigid, soft)
- Use visual cues like texture, reflectance, context

### 5. Hybrid and Multi-Task

**Superpixel + adjacent object prediction**:
- Combine irregular masking with spatial reasoning
- Predict both masked superpixel content and nearby regions
- Enhances relational reasoning

**Cross-scale consistency**:
- Ensure representations are consistent across different scales
- Use pyramid of resolutions
- Similar to multi-crop in DINO but with explicit consistency loss

**Jigsaw with semantic pieces**:
- Traditional jigsaw but pieces are superpixels or semantic regions
- More meaningful than grid-based jigsaw

## Video-Based Tasks (POTENTIALLY BEST APPROACH)

### 6. Temporal Prediction

**Frame order prediction**:
- Shuffle frames and predict correct temporal order
- Learn temporal dynamics

**Future frame prediction**:
- Predict next frame(s) from current frames
- Learn motion and dynamics
- Can be at pixel level or feature level

**Speed prediction**:
- Predict playback speed (normal, 2x, 0.5x)
- Learn temporal consistency

**Arrow of time**:
- Predict if video is playing forward or backward
- Learn natural temporal progression

### 7. Motion and Optical Flow

**Optical flow prediction**:
- Self-supervised optical flow estimation
- No ground truth labels needed
- Learn motion patterns

**Motion segmentation**:
- Segment regions with different motions
- Understand object boundaries through movement

**Ego-motion estimation**:
- Predict camera motion from video
- Useful for robotics and autonomous driving

### 8. Temporal Correspondence

**Track-before-detect**:
- Track superpixels or features across frames
- Learn object permanence and identity

**Cycle consistency**:
- Track forward then backward, should return to start
- Self-supervised constraint

**Dense correspondence**:
- Match every pixel/region across frames
- Learn spatial-temporal relationships

### 9. Video-Specific Contrastive Learning

**Temporal contrastive learning**:
- Frames from same video are positive pairs
- Frames from different videos are negative pairs
- Extension of SimCLR to temporal domain

**Slow feature analysis**:
- Features should change slowly over time
- Encourage temporal smoothness
- Contrast with SimCLR's instance discrimination

**Cross-view temporal prediction**:
- Predict one augmented view from another across time
- Combines spatial and temporal invariances

### 10. Audio-Visual Learning (Multi-Modal)

**Audio-visual correspondence**:
- Match audio to video frames
- Which object is making the sound?
- Natural supervision from synchronization

**Audio source localization**:
- Predict which region of image generates audio
- No labels needed, just synchronized data

**Cross-modal prediction**:
- Predict audio features from visual or vice versa
- Learn shared representations

## Extending SimCLR

### SimCLR Extensions for Richer Learning

**1. Temporal SimCLR**:
- Positive pairs: frames from same video clip
- Augmentations: temporal jittering, speed changes, frame sampling
- Benefits: Learn temporal invariances, motion understanding
- Use case: Video understanding, action recognition

**2. Hierarchical SimCLR**:
- Contrastive learning at multiple scales simultaneously
- Patch-level, region-level, and image-level contrasts
- Multi-scale feature pyramid
- Benefits: Capture both local and global patterns

**3. Superpixel-level SimCLR**:
- Compute contrastive loss on superpixel embeddings instead of full images
- Positive pairs: corresponding superpixels across augmentations
- Benefits: More fine-grained representations, object-part understanding

**4. Hard Negative Mining**:
- Dynamically select challenging negative pairs
- Use nearest neighbors in embedding space
- Benefits: Stronger discrimination, better feature quality

**5. Asymmetric Augmentations**:
- Different augmentation strengths for the two views
- One strong, one weak (like in FixMatch)
- Benefits: Learn invariances while preserving some visual details

**6. Cross-Modal SimCLR**:
- Positive pairs from different modalities (image-text, image-audio, image-depth)
- Learn multi-modal representations
- Benefits: Richer semantic understanding

**7. Dynamic Temperature**:
- Temperature parameter changes during training
- Start high (soft), end low (sharp)
- Benefits: Better convergence, stronger features

**8. Local-Global SimCLR**:
- Global image view + multiple local crop views
- Similar to DINO's multi-crop but with contrastive loss
- Benefits: Learn both context and details

**9. Momentum SimCLR (→ MoCo)**:
- Use momentum encoder for negatives
- Larger negative queue
- More stable training

**10. Sequential Augmentation Consistency**:
- Apply augmentations in sequence: A → B → C
- Enforce: SimCLR(A,B) + SimCLR(B,C) ≈ SimCLR(A,C)
- Benefits: More robust augmentation invariances

## Implementation Priorities

### Tier 1 (High Priority - Start Here):
1. **Superpixel masked prediction** - Multi-scale variant (8/10 confidence from consensus)
2. **Out-of-context object detection** - Requires semantic understanding, highly relevant
3. **Temporal SimCLR** - Video-based contrastive learning (user emphasized videos)
4. **Operation prediction** - Learn transformations and invariances

### Tier 2 (Medium Priority):
5. **Audio-visual correspondence** - Natural supervision from videos
6. **Spatial arrangement prediction** - Learn scene layouts
7. **Hard negative mining for SimCLR** - Improve existing method
8. **Optical flow prediction** - Self-supervised motion

### Tier 3 (Exploratory):
9. **Affordance prediction** - Semantic but challenging to self-supervise
10. **Hierarchical SimCLR** - Multi-scale contrastive
11. **Physical property prediction** - Material understanding
12. **Operation sequence prediction** - Compositional transformations

## Key Principles

1. **Perceptual alignment**: Tasks should mimic human visual processing
2. **Semantic depth**: Go beyond low-level features to understand meaning
3. **Multi-scale**: Capture hierarchical representations
4. **Temporal consistency**: Videos provide natural supervision
5. **Multi-modal**: Combine vision with audio, text, depth when available
6. **Empirical validation**: All tasks need benchmarking on downstream tasks

## Resources

- SLIC Superpixels: OpenCV implementation
- Video datasets: Kinetics, Something-Something, YouTube-8M
- Audio-visual: AudioSet, VGGSound
- Multi-modal: CLIP, ImageBind approaches

## Leveraging Existing Models for Dataset Creation

**IMPORTANT**: We should embrace using pretrained models as tools to build the datasets and pretext tasks. Don't reinvent the wheel!

**Bootstrapping Philosophy**: We may need existing models to bootstrap these techniques initially. Use pretrained models (SAM, CLIP, etc.) to create the training signal, then as our SSL models improve, we can potentially reduce reliance on the bootstrapping models. Start practical, iterate toward fully self-supervised.

### Segmentation Models

**SAM (Segment Anything Model)**:
- Use for generating high-quality segmentation masks
- Perfect for superpixel-based masking tasks
- Can create irregular region masks automatically
- Better than SLIC for semantic boundaries
- Example: `sam.predict(image)` → use masks for masking pretext task

**Semantic Segmentation Models** (DeepLab, Mask R-CNN):
- Create semantic superpixels (object-aware regions)
- Build object-scene context datasets
- Generate ground truth for spatial reasoning tasks

### Object Detection Models

**YOLO, Faster R-CNN, DETR**:
- Detect objects for out-of-context task creation
- Build spatial relationship datasets (adjacent objects)
- Create synthetic anomalies by pasting detected objects

**OWL-ViT** (Open-vocabulary detection):
- Flexible object detection with text queries
- Build diverse object datasets without fixed categories

### Multi-Modal Models

**CLIP**:
- Filter images by semantic content
- Create scene-object compatibility datasets
- Find contextually similar/dissimilar images
- Build cross-modal datasets

**ImageBind**:
- Multi-modal embeddings (vision, audio, text)
- Create audio-visual correspondence datasets
- Build multi-modal pretext tasks

### Optical Flow & Tracking

**RAFT, FlowFormer**:
- Generate optical flow pseudo-labels
- Build motion segmentation datasets
- Create temporal correspondence data

**CoTracker, TAP-Vid models**:
- Dense point tracking for videos
- Build temporal consistency datasets

### Depth Estimation

**MiDaS, DPT, Depth Anything**:
- Generate depth maps for figure-ground separation
- Create 3D reasoning datasets
- Build spatial arrangement data with depth cues

### Practical Approach

**Pipeline Example - Out-of-Context Detection**:
1. Use SAM or Mask R-CNN to segment objects
2. Use CLIP to understand scene context
3. Paste objects from incompatible scenes (beach → office)
4. Train model to detect anomalies

**Pipeline Example - Superpixel Masking**:
1. Use SAM to generate semantic masks
2. Cluster masks at different scales (multi-scale superpixels)
3. Randomly mask regions
4. Train model to predict masked content

**Pipeline Example - Temporal Correspondence**:
1. Use CoTracker to track points across video frames
2. Create positive pairs (same track) and negative pairs (different tracks)
3. Train contrastive model on correspondences

### Benefits of This Approach

- **Speed**: No manual annotation needed
- **Quality**: Pretrained models often better than hand-crafted algorithms
- **Flexibility**: Easy to experiment with different task formulations
- **Scalability**: Can process large datasets automatically
- **Semantic richness**: Modern models capture high-level semantics

### Recommended Models to Use

| Task | Recommended Model | Purpose |
|------|------------------|---------|
| Segmentation | SAM, SAM 2 | Irregular region masks |
| Object Detection | OWL-ViT, Grounding DINO | Out-of-context detection |
| Optical Flow | RAFT, FlowFormer | Temporal tasks |
| Depth | Depth Anything v2 | Spatial reasoning |
| Tracking | CoTracker | Video correspondence |
| Multi-modal | CLIP, ImageBind | Cross-modal tasks |
| Scene Understanding | CLIP, Recognize Anything | Context reasoning |

## Cautionary Notes

- **Complexity vs. Performance**: Overly complex pretexts may not transfer well
- **Computational cost**: Superpixels and video require more compute
- **Validation required**: Must test on downstream tasks (classification, detection, segmentation)
- **Dataset dependency**: Some tasks work better on certain data distributions
