"""
Superpixel-based image walking strategies.

Uses superpixel segmentation algorithms to create semantic regions,
then walks through superpixels in various orders.

=== SUPERPIXEL ALGORITHMS ===

1. **SLIC (Simple Linear Iterative Clustering)** [IMPLEMENTED]
   - Most popular, fast, well-balanced
   - K-means in 5D space (x, y, r, g, b)
   - Parameters: n_segments, compactness
   - Good for: General purpose, uniform sizes

2. **Felzenszwalb's Efficient Graph-Based Segmentation** [IMPLEMENTED]
   - Graph-based, respects boundaries well
   - Creates irregular-sized superpixels
   - Parameters: scale, sigma, min_size
   - Good for: Respecting object boundaries

3. **QuickShift** [IMPLEMENTED]
   - Mode-seeking on image + color space
   - Good boundary adherence
   - Parameters: kernel_size, max_dist, ratio
   - Good for: Texture and boundary preservation

4. **Watershed** [AVAILABLE BELOW]
   - Classic segmentation algorithm
   - Treats image as topographic surface
   - Good for: Separating touching objects

5. **Turbopixels**
   - Geometric flow-based
   - Uniform, compact superpixels
   - Good for: Regular grid-like segmentation

6. **LSC (Linear Spectral Clustering)**
   - Fast, weighted k-means in 10D space
   - Better boundary adherence than SLIC
   - Good for: High-quality segmentation

7. **SEEDS (Superpixels Extracted via Energy-Driven Sampling)**
   - Very fast, real-time capable
   - Block-based initialization
   - Good for: Speed-critical applications

8. **ERS (Entropy Rate Superpixels)**
   - Graph-based, entropy minimization
   - Excellent boundary adherence
   - Good for: Preserving fine details

9. **SCALP (Superpixels with Contour Adherence using Linear Path)**
   - Excellent boundary adherence
   - Computationally expensive
   - Good for: High-quality segmentation

10. **ETPS (Extr trusted Plane Superpixels)**
    - 3D plane fitting
    - For RGB-D images
    - Good for: When depth available

=== COMPARISON ===

| Algorithm | Speed | Boundary | Uniformity | Parameters |
|-----------|-------|----------|------------|------------|
| SLIC | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐ Good | ⭐⭐⭐⭐ High | Simple |
| Felzenszwalb | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Great | ⭐⭐ Variable | Medium |
| QuickShift | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Great | ⭐⭐ Variable | Medium |
| Watershed | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Variable | Complex |
| SEEDS | ⭐⭐⭐⭐⭐ Very Fast | ⭐⭐⭐ Good | ⭐⭐⭐⭐ High | Simple |
| LSC | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Great | ⭐⭐⭐⭐ High | Medium |

=== RECOMMENDATIONS ===

- **Start with SLIC**: Best balance of speed, quality, simplicity
- **For boundary adherence**: Felzenszwalb, Watershed, or LSC
- **For speed**: SLIC or SEEDS
- **For uniform size**: SLIC with high compactness
- **For irregular/natural**: Felzenszwalb or QuickShift
"""

from dataclasses import dataclass

import cv2
import networkx as nx
import numpy as np
from scipy.spatial import distance
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb, quickshift, slic


@dataclass
class Superpixel:
    """Information about a single superpixel."""

    id: int
    mask: np.ndarray  # Boolean mask
    center: tuple[int, int]  # Centroid (row, col)
    area: int  # Number of pixels
    mean_color: np.ndarray  # Average RGB color
    bounding_box: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)


class SuperpixelWalker:
    """Walk through image at superpixel level."""

    def __init__(
        self,
        image: np.ndarray,
        method: str = "slic",
        n_segments: int = 100,
        compactness: float = 10.0,
    ):
        """
        Args:
            image: Input image (H, W, C) or (H, W)
            method: 'slic', 'felzenszwalb', 'quickshift', or 'watershed'
            n_segments: Approximate number of superpixels
            compactness: Trade-off between color and space proximity (for SLIC)
        """
        self.image = image.copy()
        self.method = method
        self.n_segments = n_segments
        self.compactness = compactness

        # Segment image
        self.segments = self._segment_image()
        self.superpixels = self._extract_superpixels()

    def _segment_image(self) -> np.ndarray:
        """Segment image into superpixels."""
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        if self.method == "slic":
            segments = slic(
                self.image, n_segments=self.n_segments, compactness=self.compactness, start_label=0
            )

        elif self.method == "felzenszwalb":
            segments = felzenszwalb(self.image, scale=100, sigma=0.5, min_size=50)

        elif self.method == "quickshift":
            segments = quickshift(self.image, kernel_size=3, max_dist=6, ratio=0.5)

        elif self.method == "watershed":
            # Convert to grayscale if needed
            if len(self.image.shape) == 3:
                gray = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = self.image.astype(np.uint8)

            # Compute distance transform
            # First threshold to get binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            distance = ndi.distance_transform_edt(thresh)

            # Find peaks (markers)
            coords = peak_local_max(
                distance, min_distance=20, labels=thresh, num_peaks=self.n_segments
            )
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)

            # Apply watershed
            segments = watershed(-distance, markers, mask=thresh)
            segments = segments - 1  # Start from 0

        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")

        return segments

    def _extract_superpixels(self) -> dict[int, Superpixel]:
        """Extract superpixel properties."""
        superpixels = {}

        for region in regionprops(
            self.segments + 1
        ):  # +1 because regionprops expects labels starting at 1
            sp_id = region.label - 1

            # Create mask
            mask = self.segments == sp_id

            # Get properties
            center = (int(region.centroid[0]), int(region.centroid[1]))
            area = region.area
            bbox = region.bbox  # (min_row, min_col, max_row, max_col)

            # Mean color
            if len(self.image.shape) == 3:
                mean_color = np.array(
                    [self.image[:, :, c][mask].mean() for c in range(self.image.shape[2])]
                )
            else:
                mean_color = np.array([self.image[mask].mean()])

            superpixels[sp_id] = Superpixel(
                id=sp_id,
                mask=mask,
                center=center,
                area=area,
                mean_color=mean_color,
                bounding_box=bbox,
            )

        return superpixels

    def get_adjacency_graph(self) -> nx.Graph:
        """Build graph of adjacent superpixels."""
        G = nx.Graph()

        # Add all superpixels as nodes
        for sp_id in self.superpixels:
            G.add_node(sp_id)

        # Find adjacent superpixels
        h, w = self.segments.shape
        checked = set()

        for i in range(h):
            for j in range(w):
                sp_id = self.segments[i, j]

                # Check right neighbor
                if j + 1 < w:
                    neighbor_id = self.segments[i, j + 1]
                    if sp_id != neighbor_id:
                        edge = tuple(sorted([sp_id, neighbor_id]))
                        if edge not in checked:
                            G.add_edge(sp_id, neighbor_id)
                            checked.add(edge)

                # Check bottom neighbor
                if i + 1 < h:
                    neighbor_id = self.segments[i + 1, j]
                    if sp_id != neighbor_id:
                        edge = tuple(sorted([sp_id, neighbor_id]))
                        if edge not in checked:
                            G.add_edge(sp_id, neighbor_id)
                            checked.add(edge)

        return G

    def walk_by_size(self, largest_first: bool = True) -> list[int]:
        """
        Order superpixels by size.

        Args:
            largest_first: If True, largest superpixels first

        Returns:
            List of superpixel IDs in order
        """
        sizes = [(sp.id, sp.area) for sp in self.superpixels.values()]
        sizes.sort(key=lambda x: x[1], reverse=largest_first)
        return [sp_id for sp_id, _ in sizes]

    def walk_by_brightness(self, brightest_first: bool = True) -> list[int]:
        """Order superpixels by brightness."""
        brightnesses = [(sp.id, sp.mean_color.sum()) for sp in self.superpixels.values()]
        brightnesses.sort(key=lambda x: x[1], reverse=brightest_first)
        return [sp_id for sp_id, _ in brightnesses]

    def walk_by_color_variance(self, highest_first: bool = True) -> list[int]:
        """Order superpixels by color variance (uniformity)."""
        variances = []
        for sp in self.superpixels.values():
            if len(self.image.shape) == 3:
                var = np.array(
                    [self.image[:, :, c][sp.mask].var() for c in range(self.image.shape[2])]
                ).mean()
            else:
                var = self.image[sp.mask].var()
            variances.append((sp.id, var))

        variances.sort(key=lambda x: x[1], reverse=highest_first)
        return [sp_id for sp_id, _ in variances]

    def walk_by_position(self, start_corner: str = "top-left") -> list[int]:
        """
        Order superpixels by spatial position.

        Args:
            start_corner: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'
        """
        h, w = self.image.shape[:2]

        if start_corner == "top-left":
            reference = (0, 0)
        elif start_corner == "top-right":
            reference = (0, w)
        elif start_corner == "bottom-left":
            reference = (h, 0)
        elif start_corner == "bottom-right":
            reference = (h, w)
        elif start_corner == "center":
            reference = (h // 2, w // 2)
        else:
            raise ValueError(f"Unknown corner: {start_corner}")

        distances = [
            (sp.id, distance.euclidean(sp.center, reference)) for sp in self.superpixels.values()
        ]
        distances.sort(key=lambda x: x[1])
        return [sp_id for sp_id, _ in distances]

    def walk_by_gradient(self, maximize: bool = True) -> list[int]:
        """Order superpixels by gradient magnitude at boundaries."""
        from scipy import ndimage

        if len(self.image.shape) == 3:
            gray = np.sum(self.image, axis=2)
        else:
            gray = self.image

        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        gradient = np.sqrt(grad_x**2 + grad_y**2)

        # Average gradient at superpixel boundaries
        boundary_gradients = []
        for sp in self.superpixels.values():
            # Dilate mask to get boundary
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(sp.mask.astype(np.uint8), kernel, iterations=1)
            boundary = dilated.astype(bool) & ~sp.mask

            if boundary.sum() > 0:
                avg_grad = gradient[boundary].mean()
            else:
                avg_grad = 0.0

            boundary_gradients.append((sp.id, avg_grad))

        boundary_gradients.sort(key=lambda x: x[1], reverse=maximize)
        return [sp_id for sp_id, _ in boundary_gradients]

    def walk_adjacency_graph(
        self, start_id: int | None = None, strategy: str = "bfs"
    ) -> list[int]:
        """
        Walk through superpixels following adjacency graph.

        Args:
            start_id: Starting superpixel ID. If None, start from largest.
            strategy: 'bfs' (breadth-first) or 'dfs' (depth-first)

        Returns:
            List of superpixel IDs in traversal order
        """
        G = self.get_adjacency_graph()

        if start_id is None:
            # Start from largest superpixel
            start_id = max(self.superpixels.items(), key=lambda x: x[1].area)[0]

        if strategy == "bfs":
            order = list(nx.bfs_tree(G, start_id))
        elif strategy == "dfs":
            order = list(nx.dfs_tree(G, start_id))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Add any unvisited nodes
        visited = set(order)
        for node in G.nodes():
            if node not in visited:
                order.append(node)

        return order

    def visualize_superpixels(
        self, order: list[int] | None = None, show_order: bool = True
    ) -> np.ndarray:
        """
        Visualize superpixels with optional ordering.

        Args:
            order: Optional ordering of superpixels
            show_order: Show numbers indicating order

        Returns:
            Visualization image
        """
        # Create output
        if len(self.image.shape) == 2:
            output = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            output = self.image.astype(np.uint8).copy()

        # Draw superpixel boundaries
        boundaries = cv2.Canny(self.segments.astype(np.uint8), 0, 1)
        output[boundaries > 0] = [255, 255, 255]

        # If order provided, draw path
        if order is not None:
            # Draw lines between adjacent superpixels in order
            for i in range(len(order) - 1):
                sp1 = self.superpixels[order[i]]
                sp2 = self.superpixels[order[i + 1]]

                pt1 = (sp1.center[1], sp1.center[0])  # (col, row)
                pt2 = (sp2.center[1], sp2.center[0])

                # Color gradient from blue to red
                ratio = i / len(order)
                color = (int(255 * ratio), 0, int(255 * (1 - ratio)))

                cv2.line(output, pt1, pt2, color, 2)

            # Mark start and end
            start_sp = self.superpixels[order[0]]
            end_sp = self.superpixels[order[-1]]

            cv2.circle(
                output, (start_sp.center[1], start_sp.center[0]), 8, (0, 255, 0), -1
            )  # Green start
            cv2.circle(output, (end_sp.center[1], end_sp.center[0]), 8, (0, 0, 255), -1)  # Red end

            # Show order numbers
            if show_order:
                for i, sp_id in enumerate(order[:20]):  # Show first 20
                    sp = self.superpixels[sp_id]
                    cv2.putText(
                        output,
                        str(i),
                        (sp.center[1] - 10, sp.center[0] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0),
                        1,
                    )

        return output
