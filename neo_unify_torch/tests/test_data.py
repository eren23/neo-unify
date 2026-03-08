"""Tests for data generation."""

import torch
import pytest
from neo_unify_torch.shared.data import generate_dataset, CLASSES, NUM_CLASSES


def test_shape():
    images, labels = generate_dataset(samples_per_class=10)
    assert images.shape == (60, 3, 16, 16), f"Expected (60, 3, 16, 16), got {images.shape}"
    assert labels.shape == (60,), f"Expected (60,), got {labels.shape}"


def test_nchw_format():
    images, _ = generate_dataset(samples_per_class=10)
    assert images.shape[1] == 3, "Channel dim should be 3 (NCHW)"
    assert images.shape[2] == 16, "Height should be 16"
    assert images.shape[3] == 16, "Width should be 16"


def test_value_range():
    images, _ = generate_dataset(samples_per_class=100)
    assert images.min() >= 0.0, f"Min value {images.min()} < 0"
    assert images.max() <= 1.0, f"Max value {images.max()} > 1"


def test_dtype():
    images, labels = generate_dataset(samples_per_class=10)
    assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
    assert labels.dtype == torch.int64, f"Expected int64, got {labels.dtype}"


def test_class_balance():
    images, labels = generate_dataset(samples_per_class=50)
    for i in range(NUM_CLASSES):
        count = (labels == i).sum().item()
        assert count == 50, f"Class {i} has {count} samples, expected 50"


def test_determinism():
    images1, labels1 = generate_dataset(samples_per_class=10, seed=123)
    images2, labels2 = generate_dataset(samples_per_class=10, seed=123)
    assert torch.equal(images1, images2), "Same seed should produce same images"
    assert torch.equal(labels1, labels2), "Same seed should produce same labels"


def test_different_seeds():
    images1, _ = generate_dataset(samples_per_class=10, seed=1)
    images2, _ = generate_dataset(samples_per_class=10, seed=2)
    assert not torch.equal(images1, images2), "Different seeds should produce different images"


def test_all_classes_present():
    _, labels = generate_dataset(samples_per_class=10)
    unique = torch.unique(labels)
    assert len(unique) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(unique)}"
