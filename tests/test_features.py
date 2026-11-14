import numpy as np
import pytest

from features import NUM_FEATURES, landmarks_to_features


@pytest.fixture
def hand():
    return np.random.default_rng(0).uniform(0.2, 0.8, size=(21, 3))


def test_shape_and_flat_input(hand):
    features = landmarks_to_features(hand)
    assert features.shape == (NUM_FEATURES,) == (60,)
    np.testing.assert_allclose(landmarks_to_features(hand.reshape(-1)), features)


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        landmarks_to_features(np.zeros((20, 3)))


def test_translation_invariant(hand):
    moved = hand + np.array([0.15, -0.1, 0.05])
    np.testing.assert_allclose(
        landmarks_to_features(moved), landmarks_to_features(hand), atol=1e-9
    )


def test_scale_invariant(hand):
    # Scale about an arbitrary point: a hand closer to or further from the camera.
    centre = np.array([0.5, 0.5, 0.0])
    for factor in (0.3, 2.5):
        scaled = (hand - centre) * factor + centre
        np.testing.assert_allclose(
            landmarks_to_features(scaled), landmarks_to_features(hand), atol=1e-9
        )


def test_hand_size_is_one(hand):
    features = landmarks_to_features(hand).reshape(20, 3)
    assert np.linalg.norm(features[9 - 1]) == pytest.approx(1.0)  # middle-finger MCP


def test_mirror_invariant(hand):
    # A left hand is the mirror image of a right hand making the same shape.
    mirrored = hand.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    np.testing.assert_allclose(
        landmarks_to_features(mirrored, "Left"),
        landmarks_to_features(hand, "Right"),
        atol=1e-9,
    )
    # Without the handedness flag the two must differ, or the test proves nothing.
    assert not np.allclose(
        landmarks_to_features(mirrored, "Right"), landmarks_to_features(hand, "Right")
    )


def test_aspect_ratio_makes_units_equal():
    # The same physical hand, seen in a 16:9 frame: x is squeezed by 9/16.
    square = np.random.default_rng(3).uniform(0.2, 0.8, size=(21, 3))
    wide = square.copy()
    wide[:, 0] /= 16 / 9
    wide[:, 2] /= 16 / 9
    np.testing.assert_allclose(
        landmarks_to_features(wide, aspect_ratio=16 / 9),
        landmarks_to_features(square, aspect_ratio=1.0),
        atol=1e-9,
    )


def test_zero_size_hand_is_finite():
    # All landmarks on one point: wrist -> middle MCP distance is zero.
    collapsed = np.full((21, 3), 0.5)
    features = landmarks_to_features(collapsed)
    assert np.all(np.isfinite(features))
    np.testing.assert_array_equal(features, np.zeros(NUM_FEATURES))

    # Only the reference bone has zero length; the rest must stay finite.
    hand = np.random.default_rng(1).uniform(0.2, 0.8, size=(21, 3))
    hand[9] = hand[0]
    assert np.all(np.isfinite(landmarks_to_features(hand)))


def test_input_not_modified(hand):
    before = hand.copy()
    landmarks_to_features(hand, "Left", 16 / 9)
    np.testing.assert_array_equal(hand, before)
