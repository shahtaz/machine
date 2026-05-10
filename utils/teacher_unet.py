# =============================================================================
# TEACHER MODEL — U-NET + RESNET34
# CARDIAC MRI SEGMENTATION
# MODEL DEFINITION ONLY (CLEAN + FIXED)
# =============================================================================

from __future__ import annotations

import sys
import logging
import tensorflow as tf

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# SEGMENTATION MODELS
# -----------------------------------------------------------------------------
try:
    import segmentation_models as sm
    sm.set_framework("tf.keras")
except ImportError:
    log.error(
        "segmentation_models not installed.\n"
        "Run: pip install segmentation-models"
    )
    sys.exit(1)

# =============================================================================
# CONSTANTS
# =============================================================================
BACKBONE: str = "resnet34"
NUM_CLASSES: int = 4
IMG_SIZE: int = 256

__all__ = [
    "BACKBONE",
    "NUM_CLASSES",
    "IMG_SIZE",
    "preprocess_image",
    "build_teacher_model",
]

# =============================================================================
# PREPROCESSING (FIXED: COMPUTED ONCE ONLY)
# =============================================================================

_PREPROCESS_FN = sm.get_preprocessing(BACKBONE)

def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    """
    Apply ResNet34 ImageNet preprocessing.

    Args:
        image: float32 tensor (H, W, 3) in [0, 255]

    Returns:
        Preprocessed tensor
    """
    return _PREPROCESS_FN(image)

# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_teacher_model(
    img_size: int = IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    backbone: str = BACKBONE,
    encoder_freeze: bool = False,
) -> tf.keras.Model:
    """
    Teacher model: U-Net + ResNet34 encoder
    """

    model = sm.Unet(
        backbone_name=backbone,
        encoder_weights="imagenet",
        input_shape=(img_size, img_size, 3),
        classes=num_classes,
        activation="softmax",
        encoder_freeze=encoder_freeze,
        decoder_use_batchnorm=True,
    )

    log.info(
        "Teacher model built | "
        f"backbone={backbone}, "
        f"classes={num_classes}, "
        f"input=({img_size},{img_size},3), "
        f"encoder_freeze={encoder_freeze}"
    )

    return model

# =============================================================================
# SANITY CHECK
# =============================================================================
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    model = build_teacher_model()
    model.summary(line_length=120)

    # Test preprocessing
    dummy = tf.ones((IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32) * 127.0
    out = preprocess_image(dummy)

    log.info(f"Preprocess OK | shape: {out.shape}")
    log.info("Teacher model is valid and ready.")