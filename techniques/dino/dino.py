"""
DINO: Self-Distillation with No Labels
Paper: https://arxiv.org/abs/2104.14294

DINOv2: Learning Robust Visual Features without Supervision
Paper: https://arxiv.org/abs/2304.07193

Core Concepts:
- Self-distillation: Student network learns to match teacher network's outputs
- Student is updated via backprop, Teacher via exponential moving average (EMA)
- Multi-crop training: Global and local crops with different augmentations
- Centering and sharpening to avoid mode collapse
- No negative pairs needed (unlike contrastive methods)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """
    Projection head for DINO.
    Maps encoder features to a lower-dimensional space for distillation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        use_bn: bool = False,
        norm_last_layer: bool = True,
    ):
        """
        Args:
            in_dim: Input dimension from backbone
            out_dim: Output dimension (vocabulary size for distillation)
            hidden_dim: Hidden layer dimension
            bottleneck_dim: Bottleneck dimension before final layer
            use_bn: Whether to use batch normalization
            norm_last_layer: Whether to normalize the last layer weights
        """
        super().__init__()

        if use_bn:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, bottleneck_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, bottleneck_dim),
            )

        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.norm_last_layer = norm_last_layer

        if norm_last_layer:
            # Normalize weights to prevent collapse
            self.last_layer.weight.data.copy_(F.normalize(self.last_layer.weight.data, dim=1))

    def forward(self, x):
        x = self.mlp(x)
        # L2 normalization before final projection
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Wrapper to handle multiple crops at different scales.
    Performs forward pass on multiple crops and returns concatenated output.
    """

    def __init__(self, backbone, head):
        """
        Args:
            backbone: Feature encoder (e.g., Vision Transformer)
            head: Projection head (DINOHead)
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: list[torch.Tensor]):
        """
        Args:
            x: List of crops [global_crop1, global_crop2, local_crop1, ..., local_cropN]

        Returns:
            Concatenated output from all crops
        """
        # Only use global crops (first 2) for teacher

        # Forward all crops through backbone
        if not isinstance(x, list):
            x = [x]

        # Get indices for global and local crops
        torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        # Concatenate all crops
        concatenated = torch.cat(x)

        # Forward through backbone
        features = self.backbone(concatenated)

        # Forward through head
        output = self.head(features)

        return output


class DINOLoss(nn.Module):
    """
    DINO loss with centering and sharpening.
    """

    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        """
        Args:
            out_dim: Output dimension of the head
            teacher_temp: Temperature for teacher softmax (lower = sharper)
            student_temp: Temperature for student softmax
            center_momentum: EMA momentum for centering
        """
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        # Register center as buffer (not a parameter)
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor):
        """
        Cross-entropy loss between student and teacher outputs.

        Args:
            student_output: Student predictions for all crops [N_crops, batch_size, out_dim]
            teacher_output: Teacher predictions for global crops only [2, batch_size, out_dim]

        Returns:
            DINO loss value
        """
        # Normalize and apply temperature to student predictions
        student_out = student_output / self.student_temp
        student_out = F.log_softmax(student_out, dim=-1)

        # Normalize, center, and apply temperature to teacher predictions
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()  # Stop gradient

        # Compute cross-entropy loss
        # Each global view from teacher should match all crops from student
        total_loss = 0
        n_loss_terms = 0

        for teacher_view in teacher_out:
            for student_view in student_out:
                # Skip when student view matches teacher view (same augmentation)
                loss = torch.sum(-teacher_view * student_view, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center with EMA
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """
        Update center used for centering the teacher outputs.
        Prevents mode collapse by ensuring outputs are centered.

        Args:
            teacher_output: Teacher predictions
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


@torch.no_grad()
def update_teacher(student_model: nn.Module, teacher_model: nn.Module, momentum: float = 0.996):
    """
    Update teacher network weights using exponential moving average of student weights.
    This is a key component of DINO - the teacher is not trained directly.

    Args:
        student_model: Student network (trained via backprop)
        teacher_model: Teacher network (updated via EMA)
        momentum: EMA momentum (typically 0.996-0.999)
    """
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)


class DINO(nn.Module):
    """
    Complete DINO model with student and teacher networks.
    """

    def __init__(
        self,
        student_backbone,
        teacher_backbone,
        embed_dim: int = 384,
        out_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        teacher_momentum: float = 0.996,
    ):
        """
        Args:
            student_backbone: Student encoder
            teacher_backbone: Teacher encoder (typically same architecture as student)
            embed_dim: Embedding dimension from backbone
            out_dim: Output dimension for distillation
            teacher_temp: Teacher temperature
            student_temp: Student temperature
            center_momentum: Center update momentum
            teacher_momentum: Teacher EMA momentum
        """
        super().__init__()

        # Build student and teacher networks
        student_head = DINOHead(embed_dim, out_dim)
        teacher_head = DINOHead(embed_dim, out_dim)

        self.student = MultiCropWrapper(student_backbone, student_head)
        self.teacher = MultiCropWrapper(teacher_backbone, teacher_head)

        # Teacher parameters are frozen (updated via EMA)
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())

        self.criterion = DINOLoss(
            out_dim=out_dim,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )

        self.teacher_momentum = teacher_momentum

    def forward(self, crops: list[torch.Tensor]):
        """
        Forward pass with multi-crop training.

        Args:
            crops: List of image crops [global_crop1, global_crop2, local_crop1, ..., local_cropN]
                  Typically 2 global crops at 224x224 and 6-8 local crops at 96x96

        Returns:
            DINO loss
        """
        # Student processes all crops
        student_output = self.student(crops)

        # Teacher only processes global crops (first 2)
        with torch.no_grad():
            teacher_output = self.teacher(crops[:2])

        # Compute loss
        loss = self.criterion(student_output, teacher_output)

        return loss

    def update_teacher(self):
        """Update teacher network with EMA of student."""
        update_teacher(self.student, self.teacher, self.teacher_momentum)
