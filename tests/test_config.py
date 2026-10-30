"""Merkezi sınıf ve veri seti ayarlarının tutarlılık testleri."""

from __future__ import annotations

import unittest

import yaml

from config import (
    CLASS_ID_TO_NAME_TR,
    DATASET_YAML_PATH,
    DEFECT_CLASSES_TECHNICAL,
    DEFECT_CLASSES_TR,
    DEFECT_NAME_TR_BY_TECHNICAL,
    NUM_CLASSES,
    PROJECT_ROOT,
)


class ConfigTests(unittest.TestCase):
    def test_streamlit_system_dependencies_include_libgl(self) -> None:
        packages = (
            (PROJECT_ROOT / "packages.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )

        self.assertIn("libgl1", packages)
        self.assertIn("libglib2.0-0", packages)

    def test_streamlit_uses_the_opencv_variant_required_by_ultralytics(self) -> None:
        requirements = (
            (PROJECT_ROOT / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )

        self.assertIn("opencv-python==4.10.0.84", requirements)
        self.assertNotIn("opencv-python-headless==4.10.0.84", requirements)

    def test_class_lists_follow_the_dataset_order(self) -> None:
        expected_turkish = [
            DEFECT_NAME_TR_BY_TECHNICAL[class_name]
            for class_name in DEFECT_CLASSES_TECHNICAL
        ]
        self.assertEqual(DEFECT_CLASSES_TR, expected_turkish)
        self.assertEqual(list(CLASS_ID_TO_NAME_TR.values()), expected_turkish)
        self.assertEqual(NUM_CLASSES, 12)

    @unittest.skipUnless(DATASET_YAML_PATH.exists(), "Hazırlanmış dataset.yaml bulunmuyor.")
    def test_dataset_yaml_uses_the_same_class_order(self) -> None:
        payload = yaml.safe_load(DATASET_YAML_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["names"], DEFECT_CLASSES_TECHNICAL)
        self.assertEqual(payload["nc"], NUM_CLASSES)


if __name__ == "__main__":
    unittest.main()
