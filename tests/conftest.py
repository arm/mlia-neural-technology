# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest config for MLIA Neural Technology plugin tests."""

# mypy: disable-error-code=misc
import shutil
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import tensorflow as tf
import tf_keras as keras


def save_keras_model(model: keras.Model, path: Path) -> None:
    """Save a Keras model to the given path."""
    model.save(path)


def convert_to_tflite(model: keras.Model, quantized: bool, output_path: Path) -> None:
    """Convert a Keras model to a TFLite file."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def get_test_keras_model() -> keras.Model:
    """Return test Keras model."""
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1), batch_size=1, name="input"),
            keras.layers.Reshape((28, 28, 1)),
            keras.layers.Conv2D(
                filters=12, kernel_size=(3, 3), activation="relu", name="conv1"
            ),
            keras.layers.Conv2D(
                filters=12, kernel_size=(3, 3), activation="relu", name="conv2"
            ),
            keras.layers.MaxPool2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, name="output"),
        ]
    )

    model.compile(optimizer="sgd", loss="mean_squared_error")
    return model


def get_test_keras_model_no_activation() -> keras.Model:
    """Return test Keras model without activations."""
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1), batch_size=1, name="input"),
            keras.layers.Reshape((28, 28, 1)),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), name="conv1"),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), name="conv2"),
            keras.layers.MaxPool2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, name="output"),
        ]
    )

    model.compile(optimizer="sgd", loss="mean_squared_error")
    return model


TEST_MODEL_KERAS_FILE = "test_model.h5"
TEST_MODEL_TFLITE_FP32_FILE = "test_model_fp32.tflite"
TEST_MODEL_TFLITE_INT8_FILE = "test_model_int8.tflite"
TEST_MODEL_TFLITE_NO_ACT_FILE = "test_model_no_act.tflite"
TEST_MODEL_TOSA_FILE = "model.tosa"
TEST_MODEL_TOSA_MLIR_FILE = "model.tosa.mlir"
TEST_MODEL_VGF_FILE = "model.vgf"
TEST_MODEL_PTE_FILE = "model.pte"
TEST_MODEL_INVALID_FILE = "invalid.tflite"


@pytest.fixture(scope="session", name="test_models_path")
def fixture_test_models_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    """Provide path to the test models."""
    tmp_path = tmp_path_factory.mktemp("models")

    keras_model = get_test_keras_model()
    save_keras_model(keras_model, tmp_path / TEST_MODEL_KERAS_FILE)

    convert_to_tflite(
        keras_model, quantized=False, output_path=tmp_path / TEST_MODEL_TFLITE_FP32_FILE
    )

    convert_to_tflite(
        get_test_keras_model_no_activation(),
        quantized=False,
        output_path=tmp_path / TEST_MODEL_TFLITE_NO_ACT_FILE,
    )

    tflite_model_path = tmp_path / TEST_MODEL_TFLITE_INT8_FILE
    convert_to_tflite(keras_model, quantized=True, output_path=tflite_model_path)

    (tmp_path / TEST_MODEL_TOSA_FILE).write_text("tosa", encoding="utf-8")
    (tmp_path / TEST_MODEL_TOSA_MLIR_FILE).write_text("tosa", encoding="utf-8")
    (tmp_path / TEST_MODEL_VGF_FILE).write_text("vgf", encoding="utf-8")
    (tmp_path / TEST_MODEL_PTE_FILE).write_bytes(b"pte")

    invalid_tflite_model = tmp_path / TEST_MODEL_INVALID_FILE
    invalid_tflite_model.touch()

    yield tmp_path

    shutil.rmtree(tmp_path)


@pytest.fixture(scope="session", name="test_resources_path")
def fixture_test_resources_path() -> Path:
    """Return test resources path."""
    return Path(__file__).parent / "test_resources"


@pytest.fixture(scope="session", name="test_keras_model")
def fixture_test_keras_model(test_models_path: Path) -> Path:
    """Return test Keras model."""
    return test_models_path / TEST_MODEL_KERAS_FILE


@pytest.fixture(scope="session", name="test_tflite_model")
def fixture_test_tflite_model(test_models_path: Path) -> Path:
    """Return test TensorFlow Lite model."""
    return test_models_path / TEST_MODEL_TFLITE_INT8_FILE


@pytest.fixture(scope="session", name="test_tflite_model_fp32")
def fixture_test_tflite_model_fp32(test_models_path: Path) -> Path:
    """Return test TensorFlow Lite model."""
    return test_models_path / TEST_MODEL_TFLITE_FP32_FILE


@pytest.fixture(scope="session", name="test_tflite_no_act_model")
def fixture_test_tflite_no_act_model(test_models_path: Path) -> Path:
    """Return test TensorFlow Lite model with no activation."""
    return test_models_path / TEST_MODEL_TFLITE_NO_ACT_FILE


@pytest.fixture(scope="session", name="test_tosa_mlir_model")
def fixture_test_tosa_mlir_model(test_models_path: Path) -> Path:
    """Return test TOSA model."""
    return test_models_path / TEST_MODEL_TOSA_FILE


@pytest.fixture(scope="session", name="test_vgf_model")
def fixture_test_vgf_model(test_models_path: Path) -> Path:
    """Return test VGF model."""
    return test_models_path / TEST_MODEL_VGF_FILE


@pytest.fixture(scope="session", name="test_pte_model")
def fixture_test_pte_model(test_models_path: Path) -> Path:
    """Return test PTE model."""
    return test_models_path / TEST_MODEL_PTE_FILE


@pytest.fixture(scope="session", name="test_tflite_invalid_model")
def fixture_test_tflite_invalid_model(test_models_path: Path) -> Path:
    """Return test invalid TensorFlow Lite model."""
    return test_models_path / TEST_MODEL_INVALID_FILE


@pytest.fixture(scope="session", name="test_tfrecord")
def fixture_test_tfrecord(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    """Create a tfrecord with random data matching fixture 'test_tflite_model'."""

    def random_data() -> np.ndarray:
        return np.random.randint(low=-127, high=128, size=(1, 28, 28, 1), dtype=np.int8)

    tmp_path = tmp_path_factory.mktemp("tfrecords")
    tfrecord_file = tmp_path / "test.tfrecord"

    with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
        for _ in range(3):
            tensor = random_data()
            serialized = tf.io.serialize_tensor(tensor).numpy()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "serving_default_input:0": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[serialized])
                        )
                    }
                )
            )
            writer.write(example.SerializeToString())

    yield tfrecord_file

    shutil.rmtree(tmp_path)
